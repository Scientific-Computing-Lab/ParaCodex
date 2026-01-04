// #ifdef __cplusplus
// extern "C" {
// #endif

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdlib.h>									// (in directory known to compiler)			needed by malloc
#include <stdio.h>									// (in directory known to compiler)			needed by printf
#include <omp.h>

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

	long long time0 = get_time();
	int threadsPerBlock = order < 1024 ? order : 1024;
	int thread_limit = threadsPerBlock < 256 ? threadsPerBlock : 256;
	long long time1 = get_time();

	#pragma omp target data																													\
		map(to:		knodes[0:knodes_elem], start[0:count], end[0:count])																	\
		map(tofrom:	currKnode[0:count], offset[0:count], lastKnode[0:count], offset_2[0:count], recstart[0:count], reclength[0:count])
	{
		#pragma omp target teams distribute parallel for thread_limit(thread_limit)
	for (int bid = 0; bid < count; ++bid) {
			long local_curr = currKnode[bid];
			long local_offset = offset[bid];
			long local_last = lastKnode[bid];
			long local_offset2 = offset_2[bid];
			int query_start = start[bid];
			int query_end = end[bid];
			int local_recstart = recstart[bid];

			for (int level = 0; level < maxheight; ++level) {
				long next_curr = local_offset;
				long next_last = local_offset2;
				knode *node_curr = &knodes[local_curr];
				knode *node_last = &knodes[local_last];

				for (int thid = 0; thid < threadsPerBlock; ++thid) {
					int low = node_curr->keys[thid];
					int high = node_curr->keys[thid + 1];
					if (low <= query_start && high > query_start) {
						long candidate = node_curr->indices[thid];
						if (candidate < knodes_elem) {
							next_curr = candidate;
						}
						break;
					}
				}

				for (int thid = 0; thid < threadsPerBlock; ++thid) {
					int low = node_last->keys[thid];
					int high = node_last->keys[thid + 1];
					if (low <= query_end && high > query_end) {
						long candidate = node_last->indices[thid];
						if (candidate < knodes_elem) {
							next_last = candidate;
						}
						break;
					}
				}

				local_curr = next_curr;
				local_offset = next_curr;
				local_last = next_last;
				local_offset2 = next_last;
			}

			currKnode[bid] = local_curr;
			offset[bid] = local_offset;
			lastKnode[bid] = local_last;
			offset_2[bid] = local_offset2;

			knode *leaf_start = &knodes[local_curr];
			for (int thid = 0; thid < threadsPerBlock; ++thid) {
				if (leaf_start->keys[thid] == query_start) {
					local_recstart = leaf_start->indices[thid];
					recstart[bid] = local_recstart;
					break;
				}
			}

			knode *leaf_end = &knodes[local_last];
			for (int thid = 0; thid < threadsPerBlock; ++thid) {
				if (leaf_end->keys[thid] == query_end) {
					int end_idx = leaf_end->indices[thid];
					reclength[bid] = end_idx - local_recstart + 1;
					break;
				}
			}
		}
	}

	GATE_CHECKSUM_BYTES("bptree:recstart", recstart, sizeof(int)*count);
	GATE_CHECKSUM_BYTES("bptree:reclength", reclength, sizeof(int)*count);

	long long time2 = get_time();

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
