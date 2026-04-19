#include <omp.h>

#include "/root/codex_baseline/custom_serial_to_omp_workdir_20260412181326/gate_sdk/gate.h"

#define DEFAULT_ORDER 508 // Inlined from common.h

typedef struct record {
	int value;
} record;

typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER + 1];
	int  keys [DEFAULT_ORDER + 1];
	bool is_leaf;
	int num_keys;
} knode;

static inline int
find_child_slot(const knode *node, int key)
{
	int slot = 0;
	for (int i = 0; i < node->num_keys - 1; ++i) {
		if (key >= node->keys[i] && key < node->keys[i + 1]) {
			slot = i;
			break;
		}
	}
	return slot;
}

static inline int
find_leaf_slot(const knode *node, int key)
{
	for (int i = 1; i < node->num_keys - 1; ++i) {
		if (node->keys[i] == key) {
			return i;
		}
	}
	return -1;
}

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
	(void)order;
	(void)maxheight;

#pragma omp target teams distribute parallel for schedule(static) \
	map(to: knodes[0:knodes_elem], start[0:count], end[0:count]) \
	map(tofrom: currKnode[0:count], offset[0:count], lastKnode[0:count], offset_2[0:count], recstart[0:count], reclength[0:count])
	for (int bid = 0; bid < count; ++bid) {
		int start_key = start[bid];
		int end_key = end[bid];

		long start_node = 0;
		while (!knodes[start_node].is_leaf) {
			int slot = find_child_slot(&knodes[start_node], start_key);
			start_node = knodes[start_node].indices[slot];
		}

		long end_node = 0;
		while (!knodes[end_node].is_leaf) {
			int slot = find_child_slot(&knodes[end_node], end_key);
			end_node = knodes[end_node].indices[slot];
		}

		currKnode[bid] = start_node;
		offset[bid] = start_node;
		lastKnode[bid] = end_node;
		offset_2[bid] = end_node;

		recstart[bid] = 0;
		reclength[bid] = 0;

		int start_slot = find_leaf_slot(&knodes[start_node], start_key);
		int end_slot = find_leaf_slot(&knodes[end_node], end_key);
		if (start_slot >= 0) {
			recstart[bid] = knodes[start_node].indices[start_slot];
		}
		if (start_slot >= 0 && end_slot >= 0) {
			reclength[bid] = knodes[end_node].indices[end_slot] - recstart[bid] + 1;
		}
	}

	(void)cores_arg;
}
