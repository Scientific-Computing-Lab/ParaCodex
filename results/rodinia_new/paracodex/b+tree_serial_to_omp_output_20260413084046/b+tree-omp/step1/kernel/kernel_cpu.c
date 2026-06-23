#include "kernel_cpu.h"

#pragma omp declare target
static inline long find_child_index(const knode *node, int key)
{
	long idx = 0;
	while (idx < node->num_keys && key >= node->keys[idx]) {
		idx++;
	}
	return idx;
}
#pragma omp end declare target

void
kernel_cpu(int cores_arg,
		record *recordsD,
		knode *knodesD,
		long knodes_elem,
		long records_elem,
		int order,
		long height,
		int numBlocks,
		long *currKnodeD,
		long *offsetD,
		int *keysD,
		record *ansD)
{
	(void)cores_arg;
	(void)order;
	(void)height;
	(void)currKnodeD;
	(void)offsetD;
	(void)records_elem;

	#pragma omp target teams distribute parallel for map(to: knodesD[0:knodes_elem], recordsD[0:records_elem], keysD[0:numBlocks]) map(tofrom: ansD[0:numBlocks])
	for (int bid = 0; bid < numBlocks; ++bid) {
		long node_index = 0;

		for (long level = 0; level < height; ++level) {
			node_index = knodesD[node_index].indices[find_child_index(&knodesD[node_index], keysD[bid])];
		}

		ansD[bid].value = -1;
		for (int i = 0; i < knodesD[node_index].num_keys; ++i) {
			if (knodesD[node_index].keys[i] == keysD[bid]) {
				ansD[bid].value = recordsD[knodesD[node_index].indices[i]].value;
				break;
			}
		}
	}
}
