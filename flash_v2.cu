#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void forward_kernel(const float* Q, const float* K, const float* V, const int N, const int d,
                    const int Tc, const int Tr, const int Bc, const int Br, const float softmax_scale, 
		    const int n_kv_h, const int num,
                    float* l, float *m, float* O) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z; // batch and head index

    // Offset into Q,K,V,O,l,m - different for each batch and head
    int q_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh
    int kv_offset = (bx * n_kv_h * N * d) + (by / num * N * d);
    int lm_offset = (bx * gridDim.y * N) + (by * N);  // offset for l and m

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size_qo = Br * d;  // size of Qi, Oi
    int tile_size_kv = Bc * d;  // size of Kj, Vj
    float* Qi = sram;
    float* Oi = &sram[tile_size_qo];
    float* Kj = &sram[tile_size_qo * 2];
    float* Vj = &sram[tile_size_qo * 2 + tile_size_kv];
    float* S = &sram[tile_size_qo * 2 + tile_size_kv * 2];

    // Load Qi to SRAM
    for (int x = 0; x < d; x++) {
        Qi[(tx * d) + x] = Q[q_offset + (tile_size_qo * bz) + (tx * d) + x];
        Oi[(tx * d) + x] = 0; // zero
    }
    
    float row_m_prev = -INFINITY;
    float row_l_prev = 0;
    float row_m_new, row_l_new;

    for (int j = 0; j < Tc; j++)  {
        
        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[kv_offset + (tile_size_kv * j) + (tx * d) + x];
            Vj[(tx * d) + x] = V[kv_offset + (tile_size_kv * j) + (tx * d) + x];
        }
        // S = QK^T, row_m = rowmax(S)
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; y++) {
            float sum = 0;
            for (int x = 0; x < d; x++) {
                sum += Qi[(tx * d) + x] * Kj[(y * d) + x];
            }
            sum *= softmax_scale;
            S[(Bc * tx) + y] = sum;

            if (sum > row_m)
                row_m = sum;
        }
        
        // Compute new m
        row_m_new = max(row_m_prev, row_m);

        // P = exp(S - row_m), row_l = rowsum(P)
        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
            S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m_new);
            row_l += S[(Bc * tx) + y];
        }

        // Compute l
        row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + row_l;

        // Write O, l, m to HBM
        for (int x = 0; x < d; x++) {
            float pv = 0;  // Pij * Vj
            for (int y = 0; y < Bc; y++) {
                pv += S[(Bc * tx) + y] * Vj[(y * d) + x];
            }
            Oi[(tx * d) + x] = (__expf(row_m_prev - row_m_new)) * Oi[(tx * d) + x] + pv;
        }

        // Update l, m
        row_l_prev = row_l_new;
        row_m_prev = row_m_new;
    }
    for (int x = 0; x < d; x++) {
        O[q_offset + (tile_size_qo * bz) + (tx * d) + x] = 1 / row_l_new * Oi[(tx * d) + x];
    }

}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // TODO: determine Bc, Br dynamically
    const int Bc = 32; const int Br = 32;

    const int B = Q.size(0); 
    const int nqh = Q.size(1); const int nkvh = K.size(1); const int num = nqh / nkvh;
    const int N = Q.size(2); const int d = Q.size(3);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O, l, m to HBM
    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, nqh, N});
    auto m = torch::full({B, nqh, N}, -INFINITY);
    torch::Device device(torch::kCUDA);
    l = l.to(device); m = m.to(device);

    // Calculate SRAM size needed per block
    const int sram_size = (2 * Br * d * sizeof(float)) + (2 * Bc * d * sizeof(float)) + (Bc * Br * sizeof(float));
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, sram_size);

    dim3 grid_dim(B, nqh, Tr);  // batch_size x num_heads x Tr
    dim3 block_dim(Bc);  // Bc threads per block

    forward_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, nkvh, num, 
        l.data_ptr<float>(), m.data_ptr<float>(), O.data_ptr<float>()
    );
    return O;
}
