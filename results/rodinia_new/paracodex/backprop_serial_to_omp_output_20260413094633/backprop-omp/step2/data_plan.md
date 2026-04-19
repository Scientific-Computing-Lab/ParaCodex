# Backprop OpenMP Data Plan

## Goal
Offload the timed training step as one fused GPU region so the host only pays one device launch for the forward pass, error pass, and weight update sequence.

## Strategy
- Use a single fused offload unit in `bpnn_train_kernel()`.
- Flatten the pointer-to-pointer weight matrices into contiguous temporary buffers before the device launch.
- Keep the activations, deltas, and scalar errors as direct 1D arrays.
- Copy the updated flat weight buffers back into the original 2D host matrices after the device region.

## Arrays Inventory

### Direct 1D arrays
- `input_units` - read on device, bias element updated on host before the launch.
- `hidden_units` - written on device, copied back after the launch.
- `output_units` - written on device, copied back after the launch.
- `hidden_delta` - written on device, copied back after the launch.
- `output_delta` - written on device, copied back after the launch.
- `target` - read on device.

### Flattened 2D arrays
- `input_weights_flat` - device copy of `net->input_weights`.
- `hidden_weights_flat` - device copy of `net->hidden_weights`.
- `input_prev_weights_flat` - device copy of `net->input_prev_weights`.
- `hidden_prev_weights_flat` - device copy of `net->hidden_prev_weights`.

## Data Movement Strategy

### H->D Transfers
- Pack the four 2D matrices into flat host buffers.
- Map the 1D activation and delta arrays to the device for the full fused compute region.
- Map the packed weight buffers to the device for the full fused compute region.

### D->H Transfers
- Copy `hidden_units`, `output_units`, `hidden_delta`, `output_delta`, and the packed weight buffers back after the fused kernel.
- Unpack the updated flat buffers into the original host 2D matrices.

## Functions in Timed Region

### Must run on device
- `bpnn_train_kernel()` body

### Kept on host
- `bpnn_layerforward()`
- `bpnn_output_error()`
- `bpnn_hidden_error()`
- `bpnn_adjust_weights()`

## Loop Structure
- Forward pass over hidden layer: parallel over hidden units.
- Forward pass over output layer: parallel over output units.
- Output error: parallel over output units.
- Hidden error: parallel over hidden units.
- Output weight update: parallel over output units.
- Hidden weight update: parallel over hidden units.

## Correctness and Scale
- Default correctness size: `262144` input elements.
- Larger profiling size: `>= 262144` input elements, with `1,000,000` preferred if memory allows.

## Notes
- This benchmark uses pointer-to-pointer matrices, which are awkward for direct OpenMP deep copy.
- Flattening avoids repeated deep-copy bookkeeping and keeps the GPU region structurally simple for step 1.
