#include "kernel_cpu_2.h"

#pragma omp declare target
static inline long find_child_index(const knode *node, int key)
{
	long idx = 0;
	while (idx < node->num_keys && key >= node->keys[idx]) {
		idx++;
	}
	return idx;
}

static inline long find_exact_record_index(const knode *knodesD, long height, int key)
{
	long node_index = 0;

	for (long level = 0; level < height; ++level) {
		node_index = knodesD[node_index].indices[find_child_index(&knodesD[node_index], key)];
	}

	for (int i = 0; i < knodesD[node_index].num_keys; ++i) {
		if (knodesD[node_index].keys[i] == key) {
			return knodesD[node_index].indices[i];
		}
	}

	return -1;
}
#pragma omp end declare target

void
kernel_cpu_2(int cores_arg,
		knode *knodesD,
		long knodes_elem,
		int order,
		long height,
		int numBlocks,
		long *currKnodeD,
		long *offsetD,
		long *lastKnodeD,
		long *offset_2D,
		int *x_0,
		int *var_7,
		int *RecstartD,
		int *elem_1)
{
	(void)cores_arg;
	(void)order;
	(void)knodes_elem;
	(void)currKnodeD;
	(void)offsetD;
	(void)lastKnodeD;
	(void)offset_2D;

	#pragma omp target teams distribute parallel for map(to: knodesD[0:knodes_elem], x_0[0:numBlocks], var_7[0:numBlocks]) map(tofrom: RecstartD[0:numBlocks], elem_1[0:numBlocks])
	for (int elem = 0; elem < numBlocks; ++elem) {
		long start_index = find_exact_record_index(knodesD, height, x_0[elem]);
		long end_index = find_exact_record_index(knodesD, height, var_7[elem]);

		RecstartD[elem] = (int)start_index;
		if (start_index >= 0 && end_index >= start_index) {
			elem_1[elem] = (int)(end_index - start_index + 1);
		} else {
			elem_1[elem] = 0;
		}
	}
}
