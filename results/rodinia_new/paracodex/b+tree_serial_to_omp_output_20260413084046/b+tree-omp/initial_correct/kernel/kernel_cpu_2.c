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
findRangeK(	long height,

			knode *knodesD,
			long knodes_elem,

			long *x_2,
			long *v_3,
			long *data_4,
			long *offset_2D,
			int *x_0,
			int *var_7,
			int *RecstartD,
			int *elem_1,
            int numBlocks,
            int threadsPerBlock)
{
	for (int elem_6 = 0; elem_6 < numBlocks; ++elem_6) {
		for (int data_5 = 0; data_5 < threadsPerBlock; ++data_5) {
            long local_x_2 = x_2[elem_6];
            long local_v_3 = v_3[elem_6];
            long local_data_4 = data_4[elem_6];
            long local_offset_2D = offset_2D[elem_6];

            for(int i = 0; i < height; i++){
                if((knodesD[local_x_2].keys[data_5] <= x_0[elem_6]) && (knodesD[local_x_2].keys[data_5+1] > x_0[elem_6])){
                    if(knodes_elem > knodesD[local_x_2].indices[data_5]){
                        local_v_3 = knodesD[local_x_2].indices[data_5];
                    }
                }
                if((var_7[elem_6] >= knodesD[local_data_4].keys[data_5]) && (var_7[elem_6] < knodesD[local_data_4].keys[data_5+1])){
                    if(knodes_elem > knodesD[local_data_4].indices[data_5]){
                        local_offset_2D = knodesD[local_data_4].indices[data_5];
                    }
                }

                if(data_5==0){
                    x_2[elem_6] = local_v_3;
                    data_4[elem_6] = local_offset_2D;
                }
                local_x_2 = x_2[elem_6];
                local_v_3 = x_2[elem_6];
                local_data_4 = data_4[elem_6];
                local_offset_2D = data_4[elem_6];
            }

            // After the loop
            if(x_0[elem_6] == knodesD[x_2[elem_6]].keys[data_5]){
                RecstartD[elem_6] = knodesD[x_2[elem_6]].indices[data_5];
            }

            if(var_7[elem_6] == knodesD[data_4[elem_6]].keys[data_5]){
                elem_1[elem_6] = knodesD[data_4[elem_6]].indices[data_5] - RecstartD[elem_6]+1;
            }
        }
    }
}
