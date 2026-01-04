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
#include <stdio.h>									// (in directory known to compiler)			needed by printf, stderr
#include <omp.h>

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)
#include "gate.h"


//========================================================================================================================================================================================================200
//	KERNEL_CPU FUNCTION
//========================================================================================================================================================================================================200

void 
kernel_gpu(	int cores_arg,

			record *records,
			knode *knodes,
			long knodes_elem,
			long records_elem,

			int order,
			long maxheight,
			int count,

			long *currKnode,
			long *offset,
			int *keys,
			record *ans)
{


	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int max_nthreads;
	// printf("max # of threads = %d\n", max_nthreads);
	// printf("set # of threads = %d\n", cores_arg);

	int threadsPerBlock = order < 1024 ? order : 1024;
	int thread_limit = threadsPerBlock < 256 ? threadsPerBlock : 256;

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;
	int i;

	#pragma omp target data																											\
		map(to:		records[0:records_elem], knodes[0:knodes_elem], keys[0:count])												\
		map(tofrom:	currKnode[0:count], offset[0:count], ans[0:count])
	{
		#pragma omp target teams distribute parallel for thread_limit(thread_limit)
		for (bid = 0; bid < count; ++bid) {
			long local_curr = currKnode[bid];
			long local_offset = offset[bid];
			int query_key = keys[bid];

			for (i = 0; i < maxheight; ++i) {
				long next_node = local_offset;
				knode *node = &knodes[local_curr];
				for (thid = 0; thid < threadsPerBlock; ++thid) {
					int low = node->keys[thid];
					int high = node->keys[thid + 1];
					if (low <= query_key && high > query_key) {
						long candidate = node->indices[thid];
						if (candidate < knodes_elem) {
							next_node = candidate;
						}
						break;
					}
				}
				local_curr = next_node;
				local_offset = next_node;
			}

			currKnode[bid] = local_curr;
			offset[bid] = local_offset;

			knode *leaf = &knodes[local_curr];
			for (thid = 0; thid < threadsPerBlock; ++thid) {
				if (leaf->keys[thid] == query_key) {
					int record_idx = leaf->indices[thid];
					if (record_idx >= 0 && record_idx < records_elem) {
						ans[bid].value = records[record_idx].value;
					}
					break;
				}
			}
		}
	}
	GATE_CHECKSUM_BYTES("bptree:ans_gpu", ans, sizeof(record)*count);
}


void
kernel_cpu(	int cores_arg,

			record *records,
			knode *knodes,
			long knodes_elem,
			long records_elem,

			int order,
			long maxheight,
			int count,

			long *currKnode,
			long *offset,
			int *keys,
			record *ans)
{
	int threadsPerBlock = order < 1024 ? order : 1024;

	#pragma omp parallel for schedule(static)
	for (int bid = 0; bid < count; ++bid) {
		for (int level = 0; level < maxheight; ++level) {
			for (int thid = 0; thid < threadsPerBlock; ++thid) {
				if (knodes[currKnode[bid]].keys[thid] <= keys[bid] &&
					knodes[currKnode[bid]].keys[thid + 1] > keys[bid]) {
					if (knodes[offset[bid]].indices[thid] < knodes_elem) {
						offset[bid] = knodes[offset[bid]].indices[thid];
					}
				}
			}
			currKnode[bid] = offset[bid];
		}

		for (int thid = 0; thid < threadsPerBlock; ++thid) {
			if (knodes[currKnode[bid]].keys[thid] == keys[bid]) {
				ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
			}
		}
	}
	GATE_CHECKSUM_BYTES("bptree:ans_cpu", ans, sizeof(record)*count);
}
