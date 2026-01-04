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
#include <stdio.h>									// needed by printf

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)	needed by timer
#include "gate.h"

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

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	time1 = get_time();

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;

	// process number of querries
	#pragma omp target data map(to: knodes[0:knodes_elem], start[0:count], end[0:count]) \
	                        map(tofrom: currKnode[0:count], offset[0:count], lastKnode[0:count], offset_2[0:count], recstart[0:count], reclength[0:count])
	{
		#pragma omp target teams distribute parallel for thread_limit(threadsPerBlock)
		for(bid = 0; bid < count; bid++){

			const int start_key = start[bid];
			const int end_key = end[bid];
			long curr = currKnode[bid];
			long last = lastKnode[bid];

			// process levels of the tree
			for(i = 0; i < maxheight; i++){
				const knode *head = &knodes[curr];
				const knode *tail = &knodes[last];
				long candidate_curr = curr;
				long candidate_last = last;

				#pragma omp simd
				for(thid = 0; thid < threadsPerBlock; thid++){
					int start_low = head->keys[thid];
					int start_high = head->keys[thid + 1];
					if(start_low <= start_key && start_high > start_key){
						long child = head->indices[thid];
						if(child < knodes_elem){
							candidate_curr = child;
						}
					}

					int end_low = tail->keys[thid];
					int end_high = tail->keys[thid + 1];
					if(end_low <= end_key && end_high > end_key){
						long child = tail->indices[thid];
						if(child < knodes_elem){
							candidate_last = child;
						}
					}
				}

				curr = candidate_curr;
				last = candidate_last;
				offset[bid] = curr;
				currKnode[bid] = curr;
				offset_2[bid] = last;
				lastKnode[bid] = last;
			}

			const knode *start_leaf = &knodes[curr];
			int rec_index = 0;
			#pragma omp simd
			for(thid = 0; thid < threadsPerBlock; thid++){
				if(start_leaf->keys[thid] == start_key){
					rec_index = start_leaf->indices[thid];
				}
			}
			recstart[bid] = rec_index;

			const knode *end_leaf = &knodes[last];
			int range_length = 0;
			#pragma omp simd
			for(thid = 0; thid < threadsPerBlock; thid++){
				if(end_leaf->keys[thid] == end_key){
					range_length = end_leaf->indices[thid] - rec_index + 1;
				}
			}
			reclength[bid] = range_length;
		}
	}

	time2 = get_time();
	GATE_CHECKSUM_BYTES("bptree:recstart", recstart, sizeof(int)*count);
	GATE_CHECKSUM_BYTES("bptree:reclength", reclength, sizeof(int)*count);

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
