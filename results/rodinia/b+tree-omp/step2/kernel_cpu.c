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

//======================================================================================================================================================150
//	COMMON
//======================================================================================================================================================150

#include "../common.h"								// (in directory provided here)

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "../util/timer/timer.h"					// (in directory provided here)


#define GPU_DEVICE 1

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

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;

	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;
	int i;


	// process number of querries
	#pragma omp target data map(to: records[0:records_elem], knodes[0:knodes_elem], keys[0:count]) \
	                        map(tofrom: currKnode[0:count], offset[0:count], ans[0:count])
	{
		#pragma omp target teams distribute parallel for thread_limit(threadsPerBlock)
		for(bid = 0; bid < count; bid++){

			int query_key = keys[bid];
			long curr = currKnode[bid];

			// process levels of the tree
			for(i = 0; i < maxheight; i++){
				const knode *node = &knodes[curr];
				long candidate = curr;

				#pragma omp simd
				for(thid = 0; thid < threadsPerBlock; thid++){
					int key_lo = node->keys[thid];
					int key_hi = node->keys[thid + 1];
					if(key_lo <= query_key && key_hi > query_key){
						long child = node->indices[thid];
						if(child < knodes_elem){
							candidate = child;
						}
					}
				}

				curr = candidate;
				offset[bid] = curr;
				currKnode[bid] = curr;
			}

			const knode *leaf = &knodes[curr];
			int value = -1;

			#pragma omp simd
			for(thid = 0; thid < threadsPerBlock; thid++){
				if(leaf->keys[thid] == query_key){
					value = records[leaf->indices[thid]].value;
				}
			}

			ans[bid].value = value;
		}
	}

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


	//======================================================================================================================================================150
	//	MCPU SETUP
	//======================================================================================================================================================150

	int max_nthreads;
	// printf("max # of threads = %d\n", max_nthreads);
	// printf("set # of threads = %d\n", cores_arg);

	int threadsPerBlock;
	threadsPerBlock = order < 1024 ? order : 1024;


	//======================================================================================================================================================150
	//	PROCESS INTERACTIONS
	//======================================================================================================================================================150

	// private thread IDs
	int thid;
	int bid;
	int i;


	// process number of querries

	for(bid = 0; bid < count; bid++){

		// process levels of the tree
		for(i = 0; i < maxheight; i++){

			// process all leaves at each level
			for(thid = 0; thid < threadsPerBlock; thid++){

				// if value is between the two keys
				if((knodes[currKnode[bid]].keys[thid]) <= keys[bid] && (knodes[currKnode[bid]].keys[thid+1] > keys[bid])){
					// this conditional statement is inserted to avoid crush due to but in original code
					// "offset[bid]" calculated below that addresses knodes[] in the next iteration goes outside of its bounds cause segmentation fault
					// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
					if(knodes[offset[bid]].indices[thid] < knodes_elem){
						offset[bid] = knodes[offset[bid]].indices[thid];
					}
				}

			}

			// set for next tree level
			currKnode[bid] = offset[bid];

		}

		//At this point, we have a candidate leaf node which may contain
		//the target record.  Check each key to hopefully find the record
		// process all leaves at each level
		for(thid = 0; thid < threadsPerBlock; thid++){

			if(knodes[currKnode[bid]].keys[thid] == keys[bid]){
				ans[bid].value = records[knodes[currKnode[bid]].indices[thid]].value;
			}

		}

	}
}
