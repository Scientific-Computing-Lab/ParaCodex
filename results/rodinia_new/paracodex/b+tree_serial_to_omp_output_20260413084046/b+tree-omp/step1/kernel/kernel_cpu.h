#ifndef BPTREE_KERNEL_CPU_H
#define BPTREE_KERNEL_CPU_H

#include "../common.h"

void kernel_cpu(int cores_arg,
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
		record *ansD);

#endif
