#include "gnn_types.h"

// Gather features based on edge indices
extern "C" void gather(
    const float* input_features,   // [N x D] input features
    const int* edge_indices,       // [N x k] edge indices
    float* gathered_features,      // [N x k x D] output
    int N,                         // Number of nodes
    int D_dim,                     // Feature dimension
    int k                          // k-nearest neighbors
) {
    // AXI-Lite interfaces for control signals only
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=D_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=k bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // AXI Master interfaces for data pointers
    #pragma HLS INTERFACE m_axi port=input_features bundle=gmem0 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=edge_indices bundle=gmem1 max_read_burst_length=256
    #pragma HLS INTERFACE m_axi port=gathered_features bundle=gmem2 max_write_burst_length=256

    int H = (D_dim + D_H - 1) / D_H;  // Calculate number of banks needed
    int d_per_bank = D_H;

    // Process each bank of features
    Bank_Loop: for (int bank = 0; bank < H; bank++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8
        
        int bank_start = bank * d_per_bank;
        int bank_end = (bank + 1) * d_per_bank;
        if (bank_end > D_dim) bank_end = D_dim;
        int bank_dim = bank_end - bank_start;

        // Main gather for this bank
        Node_Loop: for (int i = 0; i < N; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=256 max=16384

            Neighbor_Loop: for (int j = 0; j < k; j++) {
                #pragma HLS PIPELINE II=1

                int neighbor_idx = edge_indices[i * k + j];

                // Gather all dimensions in this bank for this neighbor
                Dim_Loop: for (int d = 0; d < bank_dim; d++) {
                    #pragma HLS UNROLL factor=4

                    float feature_value = 0.0f;

                    if (neighbor_idx >= 0 && neighbor_idx < N) {
                        // Valid neighbor - gather feature
                        int src_idx = neighbor_idx * D_dim + bank_start + d;
                        feature_value = input_features[src_idx];
                    }
                    // else: Invalid neighbor - already initialized to 0.0f (padding)

                    // Store gathered feature
                    int dst_idx = (i * k + j) * D_dim + bank_start + d;
                    gathered_features[dst_idx] = feature_value;
                }
            }
        }
    }
}
