#include "gnn_types.h"

extern "C" void ffn(
    const float* input_features,
    const float* W1,
    float* output_features,
    int N,
    int D_dim
) {
    #pragma HLS INTERFACE m_axi port=input_features bundle=gmem0
    #pragma HLS INTERFACE m_axi port=W1 bundle=gmem1
    #pragma HLS INTERFACE m_axi port=output_features bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=N bundle=control
    #pragma HLS INTERFACE s_axilite port=D_dim bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int D_hidden = 4 * D_dim;

    Node_Loop: for (int i = 0; i < N; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=256 max=256
        Hidden_Loop: for (int j = 0; j < D_hidden; j++) {
            #pragma HLS PIPELINE II=1
            float acc = 0;
            for (int d = 0; d < MAX_D; d++) {
                #pragma HLS UNROLL factor=16
                acc += input_features[i * D_dim + d] * W1[j * D_dim + d];
            }
            output_features[i * D_hidden + j] = (acc > 0) ? acc : 0; // ReLU
        }
    }
}

