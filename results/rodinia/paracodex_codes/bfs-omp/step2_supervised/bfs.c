#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "../../../../gate_sdk/gate.h"
#include "../../common/rodiniaUtilFunctions.h"
//#define NUM_THREAD 4
#define OPEN

int no_of_nodes;
int edge_list_size;
FILE *fp;

//Structure to hold a node information
typedef struct Node
{
	int starting;
	int no_of_edges;
} Node;

#define bool int
#define true 1
#define false 0	

#define ERROR_THRESHOLD 0.05
#define GPU_DEVICE 1

void BFSGraph(int argc, char** argv);

static Node *d_graph_nodes      = NULL;
static int  *d_graph_edges      = NULL;
static bool *d_graph_mask       = NULL;
static bool *d_updating_graph_mask = NULL;
static bool *d_graph_visited    = NULL;
static int  *d_cost             = NULL;

static int device_initialized = 0;
static int host_device = -1;
static int target_device = -1;

static void init_device_context(void);
static void allocate_device_arrays(int node_count, int edge_count);
static void release_device_arrays(void);
static void copy_to_device(void *dst, const void *src, size_t bytes);
static void copy_from_device(void *dst, const void *src, size_t bytes);
static void run_gpu_bfs(int node_count);

void Usage(int argc, char**argv){

	fprintf(stderr,"Usage: %s <num_threads> <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	no_of_nodes=0;
	edge_list_size=0;
	BFSGraph( argc, argv);
}


void compareResults(int* h_cost, int* h_cost_gpu, int no_of_nodes) {
  int i,fail;
  fail = 0;

  // Compare C with D
  for (i=0; i<no_of_nodes; i++) {
      if (percentDiff(h_cost[i], h_cost_gpu[i]) > ERROR_THRESHOLD) {
	fail++;
      }
  }

  // print results
  printf("Non-Matching CPU-GPU Outputs Beyond Error Threshold of %4.2f Percent: %d\n", ERROR_THRESHOLD, fail);
}

////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    char *input_f;
	int	 num_omp_threads;
	
	if(argc!=3){
	Usage(argc, argv);
	exit(0);
	}
    
	num_omp_threads = atoi(argv[1]);
	if (num_omp_threads > 0) {
		omp_set_num_threads(num_omp_threads);
	}
	input_f = argv[2];
	
	printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(input_f,"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}

	int source = 0;

	fscanf(fp,"%d",&no_of_nodes);
   
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	bool *h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_mask_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_updating_graph_mask_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
	bool *h_graph_visited_gpu = (bool*) malloc(sizeof(bool)*no_of_nodes);

	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].starting = start;
		h_graph_nodes[i].no_of_edges = edgeno;
		h_graph_mask[i]=false;
		h_graph_mask_gpu[i]=false;
		h_updating_graph_mask[i]=false;
		h_updating_graph_mask_gpu[i]=false;
		h_graph_visited[i]=false;
		h_graph_visited_gpu[i]=false;
	}

	//read the source node from the file
	fscanf(fp,"%d",&source);
	// source=0; //tesing code line

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_mask_gpu[source]=true;
	h_graph_visited[source]=true;
	h_graph_visited_gpu[source]=true;

	fscanf(fp,"%d",&edge_list_size);

	int id,cost;
	int* h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i] = id;
	}

	if(fp)
	fclose(fp);    


	// allocate mem for the result on host side
	int* h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	int* h_cost_gpu = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i=0;i<no_of_nodes;i++){
		h_cost[i]=-1;
		h_cost_gpu[i]=-1;
	}
	h_cost[source]=0;
	h_cost_gpu[source]=0;
	
	printf("Start traversing the tree\n");
	double t_start, t_end;

	size_t node_bytes = sizeof(Node) * (size_t)no_of_nodes;
	size_t edge_bytes = sizeof(int) * (size_t)edge_list_size;
	size_t bool_bytes = sizeof(bool) * (size_t)no_of_nodes;
	size_t cost_bytes = sizeof(int) * (size_t)no_of_nodes;

	allocate_device_arrays(no_of_nodes, edge_list_size);
	copy_to_device(d_graph_nodes, h_graph_nodes, node_bytes);
	copy_to_device(d_graph_edges, h_graph_edges, edge_bytes);
	copy_to_device(d_graph_mask, h_graph_mask_gpu, bool_bytes);
	copy_to_device(d_updating_graph_mask, h_updating_graph_mask_gpu, bool_bytes);
	copy_to_device(d_graph_visited, h_graph_visited_gpu, bool_bytes);
	copy_to_device(d_cost, h_cost_gpu, cost_bytes);

	t_start = rtclock();
	run_gpu_bfs(no_of_nodes);
	t_end = rtclock();
  	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	copy_from_device(h_cost_gpu, d_cost, cost_bytes);
	release_device_arrays();

	int tid;
	bool stop;

	t_start = rtclock();
	//CPU
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;

		for(tid = 0; tid < no_of_nodes; tid++ )
		{
			if (h_graph_mask[tid] == true){ 
			h_graph_mask[tid]=false;
			for(int i=h_graph_nodes[tid].starting; i<(h_graph_nodes[tid].no_of_edges + h_graph_nodes[tid].starting); i++)
				{
				int id = h_graph_edges[i];
				if(!h_graph_visited[id])
					{
					h_cost[id]=h_cost[tid]+1;
					h_updating_graph_mask[id]=true;
					}
				}
			}
		}

  		for(int tid=0; tid< no_of_nodes ; tid++ )
		{
			if (h_updating_graph_mask[tid] == true){
			h_graph_mask[tid]=true;
			h_graph_visited[tid]=true;
			stop=true;
			h_updating_graph_mask[tid]=false;
			}
		}
	}
	while(stop);
	t_end = rtclock();
  	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(h_cost, h_cost_gpu, no_of_nodes);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	GATE_CHECKSUM_BYTES("bfs.h_cost", h_cost, sizeof(int) * no_of_nodes);
	GATE_CHECKSUM_BYTES("bfs.h_cost_gpu", h_cost_gpu, sizeof(int) * no_of_nodes);


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_graph_mask_gpu);
	free( h_updating_graph_mask);
	free( h_updating_graph_mask_gpu);
	free( h_graph_visited);
	free( h_graph_visited_gpu);
	free( h_cost);
	free( h_cost_gpu);

}

////////////////////////////////////////////////////////////////////////////////
// Device helpers
////////////////////////////////////////////////////////////////////////////////
static void init_device_context(void)
{
	if (device_initialized)
		return;

	host_device = omp_get_initial_device();
	target_device = omp_get_default_device();
	int num_devices = omp_get_num_devices();
	if (target_device == host_device && num_devices > 0) {
		target_device = host_device + 1;
	}

	device_initialized = 1;
}

static void allocate_device_arrays(int node_count, int edge_count)
{
	init_device_context();
	int device = target_device;
	d_graph_nodes = (Node*) omp_target_alloc(sizeof(Node) * node_count, device);
	d_graph_edges = (int*) omp_target_alloc(sizeof(int) * edge_count, device);
	d_graph_mask = (bool*) omp_target_alloc(sizeof(bool) * node_count, device);
	d_updating_graph_mask = (bool*) omp_target_alloc(sizeof(bool) * node_count, device);
	d_graph_visited = (bool*) omp_target_alloc(sizeof(bool) * node_count, device);
	d_cost = (int*) omp_target_alloc(sizeof(int) * node_count, device);
}

static void release_device_arrays(void)
{
	if (!device_initialized)
		return;

	int device = target_device;
	if (d_graph_nodes) {
		omp_target_free(d_graph_nodes, device);
		d_graph_nodes = NULL;
	}
	if (d_graph_edges) {
		omp_target_free(d_graph_edges, device);
		d_graph_edges = NULL;
	}
	if (d_graph_mask) {
		omp_target_free(d_graph_mask, device);
		d_graph_mask = NULL;
	}
	if (d_updating_graph_mask) {
		omp_target_free(d_updating_graph_mask, device);
		d_updating_graph_mask = NULL;
	}
	if (d_graph_visited) {
		omp_target_free(d_graph_visited, device);
		d_graph_visited = NULL;
	}
	if (d_cost) {
		omp_target_free(d_cost, device);
		d_cost = NULL;
	}
}

static void copy_to_device(void *dst, const void *src, size_t bytes)
{
	init_device_context();
	omp_target_memcpy(dst, src, bytes, 0, 0, target_device, host_device);
}

static void copy_from_device(void *dst, const void *src, size_t bytes)
{
	init_device_context();
	omp_target_memcpy(dst, src, bytes, 0, 0, host_device, target_device);
}

static void run_gpu_bfs(int node_count)
{
	int loop_stop;
	do {
		#pragma omp target teams distribute parallel for thread_limit(128) is_device_ptr( \
		    d_graph_mask, d_graph_nodes, d_graph_edges, d_graph_visited, d_updating_graph_mask, d_cost)
		for (int tid = 0; tid < node_count; tid++) {
			if (d_graph_mask[tid]) {
				d_graph_mask[tid] = false;
				const Node node = d_graph_nodes[tid];
				const int start = node.starting;
				const int end = start + node.no_of_edges;
				const int next_cost = d_cost[tid] + 1;
				for (int i = start; i < end; i++) {
					const int id = d_graph_edges[i];
					if (!d_graph_visited[id]) {
						d_cost[id] = next_cost;
						d_updating_graph_mask[id] = true;
					}
				}
			}
		}

		loop_stop = 0;
		#pragma omp target teams distribute parallel for thread_limit(128) reduction(|:loop_stop) is_device_ptr( \
		    d_updating_graph_mask, d_graph_mask, d_graph_visited)
		for (int tid = 0; tid < node_count; tid++) {
			if (d_updating_graph_mask[tid]) {
				d_graph_mask[tid] = true;
				d_graph_visited[tid] = true;
				loop_stop |= 1;
				d_updating_graph_mask[tid] = false;
			}
		}
	} while (loop_stop);
}
