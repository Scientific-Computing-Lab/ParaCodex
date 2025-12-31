// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdlib.h>									// (in directory known to compiler)

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)	needed by timer

//======================================================================================================================================================150
//	HEADER
//======================================================================================================================================================150

#include "./kernel_cpu_2.h"							// (in directory provided here)

//========================================================================================================================================================================================================200
//	PLASMAKERNEL_GPU
//========================================================================================================================================================================================================200

void 
kernel_cpu_2(	int cores_arg,

				knode *knodes,
				long knodes_elem,

				int order,
				long maxheight,
				int count,

				long *currKnode,
				long *offset,
				long *lastKnode,
				long *offset_2,
				int *start,
				int *end,
				int *recstart,
				int *reclength)
{

	//======================================================================================================================================================150
	//	Variables
	//======================================================================================================================================================150

	// timer
	long long time0;
	long long time1;
	long long time2;

	// common variables
	int i;

	time0 = get_time();

	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int max_nthreads;
	// printf("max # of threads = %d\n", max_nthreads);
	// printf("set # of threads = %d\n", cores_arg);

	int threadsPerBlock = order < 1024 ? order : 1024;
	// align threads per team with node fanout to maximize occupancy

	time1 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;

	// process number of querries
	#pragma omp target data \
		map(to: knodes[0:knodes_elem], knodes_elem, maxheight, count, start[0:count], end[0:count]) \
		map(tofrom: currKnode[0:count], offset[0:count], lastKnode[0:count], offset_2[0:count], recstart[0:count], reclength[0:count])
	{
		#pragma omp target teams loop
		for(bid = 0; bid < count; bid++){
			int query_start = start[bid];
			int query_end = end[bid];

			// process levels of the tree
			for(i = 0; i < maxheight; i++){
				long curr_node_idx = currKnode[bid];
				long last_node_idx = lastKnode[bid];
				knode *curr_node = &knodes[curr_node_idx];
				knode *last_node = &knodes[last_node_idx];

				// process all leaves at each level
				#pragma omp loop
				for(thid = 0; thid < threadsPerBlock; thid++){

					int key_low = curr_node->keys[thid];
					int key_high = curr_node->keys[thid+1];
					if(key_low <= query_start && key_high > query_start){
						long child = curr_node->indices[thid];
						// guard the child index from escaping the packed tree buffer
						if(child < knodes_elem){
							offset[bid] = child;
						}
					}

					key_low = last_node->keys[thid];
					key_high = last_node->keys[thid+1];
					if(key_low <= query_end && key_high > query_end){
						long child = last_node->indices[thid];
						// guard the child index from escaping the packed tree buffer
						if(child < knodes_elem){
							offset_2[bid] = child;
						}
					}

				}

				// set for next tree level
				currKnode[bid] = offset[bid];
				lastKnode[bid] = offset_2[bid];

			}

			// process leaves
			knode *start_leaf = &knodes[currKnode[bid]];
			#pragma omp loop
			for(thid = 0; thid < threadsPerBlock; thid++){

				// Find the index of the starting record
				if(start_leaf->keys[thid] == query_start){
					recstart[bid] = start_leaf->indices[thid];
				}

			}

			// process leaves
			knode *end_leaf = &knodes[lastKnode[bid]];
			#pragma omp loop
			for(thid = 0; thid < threadsPerBlock; thid++){

				// Find the index of the ending record
				if(end_leaf->keys[thid] == query_end){
					reclength[bid] = end_leaf->indices[thid] - recstart[bid]+1;
				}

			}

		}
	}

	time2 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	printf("Time spent in different stages of CPU/MCPU KERNEL:\n");

	printf("%15.12f s, %15.12f % : MCPU: SET DEVICE\n",					(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time2-time0) * 100);
	printf("%15.12f s, %15.12f % : CPU/MCPU: KERNEL\n",					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time2-time0) * 100);

	printf("Total time:\n");
	printf("%.12f s\n", 												(float) (time2-time0) / 1000000);

} // main

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

// #ifdef __cplusplus
// }
// #endif
