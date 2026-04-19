#include "kernel_cpu.h"

#pragma omp declare target
static inline long find_child_index(const knode *node, int key)
{
	long left = 0;
	long right = node->num_keys;

	while (left < right) {
		long mid = left + ((right - left) >> 1);
		if (key >= node->keys[mid]) {
			left = mid + 1;
		} else {
			right = mid;
		}
	}

	return left;
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
		const int query_key = keysD[bid];
		long node_index = 0;

		for (long level = 0; level < height; ++level) {
			const knode *node = &knodesD[node_index];
			node_index = node->indices[find_child_index(node, query_key)];
		}

		ansD[bid].value = -1;
		{
			const knode *leaf = &knodesD[node_index];
			for (int i = 0; i < leaf->num_keys; ++i) {
				if (leaf->keys[i] == query_key) {
					ansD[bid].value = recordsD[leaf->indices[i]].value;
					break;
				}
			}
		}
	}
}
