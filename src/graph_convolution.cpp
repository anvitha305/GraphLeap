#include "gnn_types.h"

extern "C" void graph_convolution(
    const float* node_features,
    const float* neighbor_features,
    const float* weights,
    float* output,
    int N, 
    int k, 
    int D_dim
) {
    #pragma HLS INTERFACE m_axi port=node_features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=neighbor_features bundle=gmem1
    #pragma HLS INTERFACE m_axi port=weights bundle=gmem2
    #pragma HLS INTERFACE m_axi port=output bundle=gmem3
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=k bundle=control
    #pragma HLS INTERFACE s_axilite port=D_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    Node_Loop: for (int i = 0; i < N; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=256 max=256
        
        Dim_Loop: for (int d = 0; d < D_dim; d++) {
            #pragma HLS PIPELINE II=1
            float center_feat = node_features[i * D_dim + d];
            float max_val = -1e38f;

            // Max-Relative Aggregation
            for (int j = 0; j < k; j++) {
                float rel = neighbor_features[(i * k + j) * D_dim + d] - center_feat;
                if (rel > max_val) max_val = rel;
            }
            // Apply weight
            output[i * D_dim + d] = max_val * weights[d];
        }
    }
}
