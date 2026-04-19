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

void
kernel_cpu(	int cores_arg,

			record *records,
			knode *knodes,
			long knodes_elem,

			int order,
			long maxheight,
			int count,

			long *currKnode,
			long *offset,
			int *keys,
			record *ans)
{
	(void)order;
	(void)maxheight;
	(void)records;

#pragma omp target teams distribute parallel for schedule(static) \
	map(to: knodes[0:knodes_elem], keys[0:count]) \
	map(tofrom: currKnode[0:count], offset[0:count], ans[0:count])
	for (int bid = 0; bid < count; ++bid) {
		long node_idx = 0;
		int key = keys[bid];

		while (!knodes[node_idx].is_leaf) {
			int slot = find_child_slot(&knodes[node_idx], key);
			node_idx = knodes[node_idx].indices[slot];
		}

		currKnode[bid] = node_idx;
		offset[bid] = node_idx;
		ans[bid].value = -1;

		for (int i = 1; i < knodes[node_idx].num_keys - 1; ++i) {
			if (knodes[node_idx].keys[i] == key) {
				ans[bid].value = key;
				break;
			}
		}
	}

	(void)cores_arg;
}
