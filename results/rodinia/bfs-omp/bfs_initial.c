#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include "../../common/rodiniaUtilFunctions.h"
#include "../../../gate_sdk/gate.h"

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
//Apply BFS on a Graph using OpenMP target offload
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
	
	int k=0;
	int stop_gpu = 0;
	bool stop_cpu;
	double t_start, t_end;

	t_start = rtclock();
	//GPU BFS with OpenMP target offload
	#pragma omp target data \
	    map(to: h_graph_nodes[:no_of_nodes], h_graph_edges[:edge_list_size]) \
	    map(tofrom: h_graph_mask_gpu[:no_of_nodes], h_updating_graph_mask_gpu[:no_of_nodes], h_graph_visited_gpu[:no_of_nodes], h_cost_gpu[:no_of_nodes])
	{
		do
		{
			stop_gpu = 0;

			#pragma omp target teams loop thread_limit(128)
			for(int tid = 0; tid < no_of_nodes; tid++ )
			{
				if (h_graph_mask_gpu[tid] == true){ 
					h_graph_mask_gpu[tid]=false;
					Node current = h_graph_nodes[tid];
					int edge_start = current.starting;
					int edge_end = current.starting + current.no_of_edges;
					for(int i=edge_start; i<edge_end; i++)
					{
						int id = h_graph_edges[i];
						if(!h_graph_visited_gpu[id])
						{
							h_cost_gpu[id]=h_cost_gpu[tid]+1;
							h_updating_graph_mask_gpu[id]=true;
						}
					}
				}
			}

			#pragma omp target teams loop map(tofrom: stop_gpu) thread_limit(128)
			for(int tid=0; tid< no_of_nodes ; tid++ )
			{
				if (h_updating_graph_mask_gpu[tid] == true){
					h_graph_mask_gpu[tid]=true;
					h_graph_visited_gpu[tid]=true;
					#pragma omp atomic write
					stop_gpu = 1;
					h_updating_graph_mask_gpu[tid]=false;
				}
			}

			k++;
		}
		while(stop_gpu);
	}
	t_end = rtclock();
  	fprintf(stdout, "GPU Runtime: %0.6lfs\n", t_end - t_start);

	t_start = rtclock();
	//CPU
	do
	{
		//if no thread changes this value then the loop stops
		stop_cpu=false;

		for(int tid = 0; tid < no_of_nodes; tid++ )
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
				stop_cpu=true;
				h_updating_graph_mask[tid]=false;
			}
		}
		k++;
	}
	while(stop_cpu);
	t_end = rtclock();
  	fprintf(stdout, "CPU Runtime: %0.6lfs\n", t_end - t_start);

	compareResults(h_cost, h_cost_gpu, no_of_nodes);
	GATE_CHECKSUM_U32("bfs:cost_cpu", h_cost, no_of_nodes);
	GATE_CHECKSUM_U32("bfs:cost_gpu", h_cost_gpu, no_of_nodes);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");


	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_visited);
	free( h_cost);
	free( h_graph_mask_gpu);
	free( h_updating_graph_mask_gpu);
	free( h_graph_visited_gpu);
	free( h_cost_gpu);

}
