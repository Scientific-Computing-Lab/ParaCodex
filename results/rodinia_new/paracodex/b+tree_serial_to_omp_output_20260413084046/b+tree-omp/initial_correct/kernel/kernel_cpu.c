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

void
findK(	long height,
		knode *knodesD,
		long knodes_elem,
		record *recordsD,

		long *currKnodeD,
		long *offsetD,
		int *keysD,
		record *ansD,
        int numBlocks,
        int threadsPerBlock)
{
	for (int bid = 0; bid < numBlocks; ++bid) {
		for (int thid = 0; thid < threadsPerBlock; ++thid) {
            long local_currKnode = currKnodeD[bid];
            long local_offset = offsetD[bid];

            for(int i = 0; i < height; i++){
                if(keysD[bid] >= (knodesD[local_currKnode].keys[thid]) && (knodesD[local_currKnode].keys[thid+1] > keysD[bid])){
                    if(knodes_elem > knodesD[local_offset].indices[thid]){
                        local_offset = knodesD[local_offset].indices[thid];
                    }
                }

                if(0 == thid){
                    currKnodeD[bid] = local_offset;
                }
                local_currKnode = currKnodeD[bid];
                local_offset = currKnodeD[bid];
            }

            // After the loop, the final key comparison
            if(knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]){
                ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
            }
        }
    }
}
