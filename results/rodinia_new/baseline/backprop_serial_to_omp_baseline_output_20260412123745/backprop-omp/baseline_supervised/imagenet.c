
#include <stdio.h>
#include <stdlib.h>
#include "backprop.h"
#include "gate.h"

extern int layer_size;

void load(BPNN *net)
{
  float *units;
  int nr, i, k;

  nr = layer_size;
  units = net->input_units;

  units[0] = 1.0f;
  k = 1;
  for (i = 0; i < nr; i++) {
	  units[k] = (float) rand()/RAND_MAX ;
	  k++;
    }

  GATE_CHECKSUM_BYTES("loaded_input_units", units + 1, (size_t)nr * sizeof(float));
}
