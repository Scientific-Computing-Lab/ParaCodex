#ifndef BPTREE_KERNEL_CPU_2_H
#define BPTREE_KERNEL_CPU_2_H

#include "../common.h"

void kernel_cpu_2(int cores_arg,
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
		int *elem_1);

#endif
