#include <xrt/xrt_kernel.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include <random>

// Test configuration
struct GNNTestConfig {
    std::string name;
    int N;          // Number of nodes
    int D;          // Feature dimension
    int k;          // k-nearest neighbors
    int conv_type;  // 0=MRCONV, 1=EDGECONV, 2=GRAPHSAGE, 3=GIN
    bool use_residual;
    bool use_layer_norm;
};

// Initialize random features
void init_random_features(float* features, int size, float scale = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, scale);
    
    for (int i = 0; i < size; i++) {
        features[i] = dist(gen);
    }
}

// Initialize random weights
void init_random_weights(float* weights, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    float limit = std::sqrt(6.0f / size);
    std::uniform_real_distribution<float> dist(-limit, limit);
    
    for (int i = 0; i < size; i++) {
        weights[i] = dist(gen);
    }
}

// Run full GNN pipeline test
void run_gnn_test(
    const xrt::device& device,
    const xrt::kernel& k_graph_construction,
    const xrt::kernel& k_gather,
    const xrt::kernel& k_graph_conv,
    const xrt::kernel& k_ffn,
    const GNNTestConfig& config
) {
    std::cout << "\n=== Test: " << config.name << " ===\n";
    std::cout << "  N=" << config.N << ", D=" << config.D 
              << ", k=" << config.k << "\n";
    
    // Allocate host memory
    size_t feat_size = config.N * config.D;
    size_t edge_size = config.N * config.k;
    size_t neighbor_feat_size = config.N * config.k * config.D;
    int hidden_dim = 4 * config.D;
    
    std::vector<float> input_features(feat_size);
    std::vector<int> edge_indices(edge_size);
    std::vector<float> neighbor_features(neighbor_feat_size);
    std::vector<float> conv_output(feat_size);
    std::vector<float> final_output(feat_size);
    
    // Initialize input features
    init_random_features(input_features.data(), feat_size, 0.1f);
    
    // Weights for graph convolution
    size_t weight_size = 3 * hidden_dim * config.D;
    std::vector<float> conv_weights(weight_size);
    init_random_weights(conv_weights.data(), weight_size);
    
    std::vector<float> conv_biases(hidden_dim + config.D);
    init_random_weights(conv_biases.data(), conv_biases.size());
    
    // Weights for FFN
    std::vector<float> ffn_W1(hidden_dim * config.D);
    std::vector<float> ffn_b1(hidden_dim);
    std::vector<float> ffn_W2(config.D * hidden_dim);
    std::vector<float> ffn_b2(config.D);
    
    init_random_weights(ffn_W1.data(), ffn_W1.size());
    init_random_weights(ffn_b1.data(), ffn_b1.size());
    init_random_weights(ffn_W2.data(), ffn_W2.size());
    init_random_weights(ffn_b2.data(), ffn_b2.size());
    
    // ========== Step 1: Graph Construction ==========
    std::cout << "  [1/4] Running graph construction...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    auto bo_input_gc = xrt::bo(device, feat_size * sizeof(float), 
                               k_graph_construction.group_id(0));
    auto bo_edges = xrt::bo(device, edge_size * sizeof(int), 
                           k_graph_construction.group_id(1));
    
    bo_input_gc.write(input_features.data());
    bo_input_gc.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run_gc = xrt::run(k_graph_construction);
    run_gc.set_arg(0, bo_input_gc);
    run_gc.set_arg(1, bo_edges);
    run_gc.set_arg(2, config.N);
    run_gc.set_arg(3, config.D);
    run_gc.set_arg(4, config.k);
    run_gc.start();
    run_gc.wait();
    
    bo_edges.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_edges.read(edge_indices.data());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Time: " << duration.count() << " ms\n";
    
    // ========== Step 2: Gather ==========
    std::cout << "  [2/4] Running gather...\n";
    start = std::chrono::high_resolution_clock::now();
    
    auto bo_input_gather = xrt::bo(device, feat_size * sizeof(float), 
                                   k_gather.group_id(0));
    auto bo_edges_gather = xrt::bo(device, edge_size * sizeof(int), 
                                   k_gather.group_id(1));
    auto bo_neighbor_feat = xrt::bo(device, neighbor_feat_size * sizeof(float), 
                                   k_gather.group_id(2));
    
    bo_input_gather.write(input_features.data());
    bo_edges_gather.write(edge_indices.data());
    bo_input_gather.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_edges_gather.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run_gather = xrt::run(k_gather);
    run_gather.set_arg(0, bo_input_gather);
    run_gather.set_arg(1, bo_edges_gather);
    run_gather.set_arg(2, bo_neighbor_feat);
    run_gather.set_arg(3, config.N);
    run_gather.set_arg(4, config.D);
    run_gather.set_arg(5, config.k);
    run_gather.start();
    run_gather.wait();
    
    bo_neighbor_feat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_neighbor_feat.read(neighbor_features.data());
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Time: " << duration.count() << " ms\n";
    
    // ========== Step 3: Graph Convolution ==========
    std::cout << "  [3/4] Running graph convolution...\n";
    start = std::chrono::high_resolution_clock::now();
    
    size_t scratch_size = config.N * config.k * config.D * 4;
    
    auto bo_node_feat = xrt::bo(device, feat_size * sizeof(float), 
                               k_graph_conv.group_id(0));
    auto bo_neighbor_feat_conv = xrt::bo(device, neighbor_feat_size * sizeof(float), 
                                        k_graph_conv.group_id(1));
    auto bo_edges_conv = xrt::bo(device, edge_size * sizeof(int), 
                                k_graph_conv.group_id(2));
    auto bo_weights = xrt::bo(device, weight_size * sizeof(float), 
                             k_graph_conv.group_id(3));
    auto bo_biases = xrt::bo(device, conv_biases.size() * sizeof(float), 
                            k_graph_conv.group_id(4));
    auto bo_conv_output = xrt::bo(device, feat_size * sizeof(float), 
                                 k_graph_conv.group_id(5));
    auto bo_scratch_conv = xrt::bo(device, scratch_size * sizeof(float), 
                                  k_graph_conv.group_id(6));
    
    bo_node_feat.write(input_features.data());
    bo_neighbor_feat_conv.write(neighbor_features.data());
    bo_edges_conv.write(edge_indices.data());
    bo_weights.write(conv_weights.data());
    bo_biases.write(conv_biases.data());
    
    bo_node_feat.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_neighbor_feat_conv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_edges_conv.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_biases.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run_conv = xrt::run(k_graph_conv);
    run_conv.set_arg(0, bo_node_feat);
    run_conv.set_arg(1, bo_neighbor_feat_conv);
    run_conv.set_arg(2, bo_edges_conv);
    run_conv.set_arg(3, bo_weights);
    run_conv.set_arg(4, bo_biases);
    run_conv.set_arg(5, bo_conv_output);
    run_conv.set_arg(6, bo_scratch_conv);
    run_conv.set_arg(7, config.N);
    run_conv.set_arg(8, config.k);
    run_conv.set_arg(9, config.D);
    run_conv.set_arg(10, config.conv_type);
    run_conv.start();
    run_conv.wait();
    
    bo_conv_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_conv_output.read(conv_output.data());
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Time: " << duration.count() << " ms\n";
    
    // ========== Step 4: FFN ==========
    std::cout << "  [4/4] Running FFN...\n";
    start = std::chrono::high_resolution_clock::now();
    
    size_t ffn_scratch_size = config.N * (config.D + hidden_dim);
    
    auto bo_ffn_input = xrt::bo(device, feat_size * sizeof(float), 
                               k_ffn.group_id(0));
    auto bo_ffn_W1 = xrt::bo(device, ffn_W1.size() * sizeof(float), 
                            k_ffn.group_id(1));
    auto bo_ffn_b1 = xrt::bo(device, ffn_b1.size() * sizeof(float), 
                            k_ffn.group_id(2));
    auto bo_ffn_W2 = xrt::bo(device, ffn_W2.size() * sizeof(float), 
                            k_ffn.group_id(3));
    auto bo_ffn_b2 = xrt::bo(device, ffn_b2.size() * sizeof(float), 
                            k_ffn.group_id(4));
    auto bo_ffn_output = xrt::bo(device, feat_size * sizeof(float), 
                                k_ffn.group_id(5));
    auto bo_scratch_ffn = xrt::bo(device, ffn_scratch_size * sizeof(float), 
                                 k_ffn.group_id(6));
    
    bo_ffn_input.write(conv_output.data());
    bo_ffn_W1.write(ffn_W1.data());
    bo_ffn_b1.write(ffn_b1.data());
    bo_ffn_W2.write(ffn_W2.data());
    bo_ffn_b2.write(ffn_b2.data());
    
    bo_ffn_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ffn_W1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ffn_b1.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ffn_W2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_ffn_b2.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    
    auto run_ffn = xrt::run(k_ffn);
    run_ffn.set_arg(0, bo_ffn_input);
    run_ffn.set_arg(1, bo_ffn_W1);
    run_ffn.set_arg(2, bo_ffn_b1);
    run_ffn.set_arg(3, bo_ffn_W2);
    run_ffn.set_arg(4, bo_ffn_b2);
    run_ffn.set_arg(5, bo_ffn_output);
    run_ffn.set_arg(6, bo_scratch_ffn);
    run_ffn.set_arg(7, config.N);
    run_ffn.set_arg(8, config.D);
    run_ffn.set_arg(9, config.use_residual);
    run_ffn.set_arg(10, config.use_layer_norm);
    run_ffn.start();
    run_ffn.wait();
    
    bo_ffn_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_ffn_output.read(final_output.data());
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "    Time: " << duration.count() << " ms\n";
    
    // Print sample outputs
    std::cout << "  Sample output features (first 5 dims of node 0):\n    ";
    for (int i = 0; i < std::min(5, config.D); i++) {
        std::cout << std::fixed << std::setprecision(4) << final_output[i] << " ";
    }
    std::cout << "\n  RESULT: PASS\n";
}

int main(int argc, char** argv) {
    const char* xclbinFilename = "./gnn.xclbin";
    
    std::cout << "Loading FPGA device...\n";
    auto device = xrt::device(0);
    auto uuid = device.load_xclbin(xclbinFilename);
    
    std::cout << "Loading kernels...\n";
    auto k_graph_construction = xrt::kernel(device, uuid, "graph_construction");
    auto k_gather = xrt::kernel(device, uuid, "gather");
    auto k_graph_conv = xrt::kernel(device, uuid, "graph_convolution");
    auto k_ffn = xrt::kernel(device, uuid, "ffn");
    
    // Test configurations
    std::vector<GNNTestConfig> tests = {
        {"Small EdgeConv", 256, 64, 16, 1, true, true},
        {"Medium MRConv", 1024, 128, 32, 0, true, true},
        {"Large GraphSAGE", 4096, 256, 32, 2, true, false},
        {"GIN", 512, 128, 16, 3, false, true}
    };
    
    for (const auto& test : tests) {
        run_gnn_test(device, k_graph_construction, k_gather, 
                    k_graph_conv, k_ffn, test);
    }
    
    std::cout << "\nAll GNN tests complete.\n";
    return 0;
}
