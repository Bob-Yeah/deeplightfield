#include "Encoder.h"
#include "thread_index.h"

/// idx3.y = 0: x, y, z, sin(x), sin(y), sin(z), cos(x), cos(y), cos(z)
/// idx3.y = 1: sin(2x), sin(2y), sin(2z), cos(2x), cos(2y), cos(2z)
/// ...
/// idx3.y = n_freq-1: sin(2^(n_freq-1)x), sin(2^(n_freq-1)y), sin(2^(n_freq-1)z),
///                    cos(2^(n_freq-1)x), cos(2^(n_freq-1)y), cos(2^(n_freq-1)z)
/// Dispatch (n_freq, n_batch, 1)
__global__ void cu_encode(glm::vec3 *o_encoded, glm::vec3 *input, float *freqs, uint n)
{
    glm::uvec3 idx3 = IDX3;
    if (idx3.y >= n)
        return;
    uint encode_dim = blockDim.x * 2 + 1;
    uint offset = idx3.y * encode_dim;
    if (idx3.x == 0)
        o_encoded[offset] = input[idx3.y];
    glm::vec3 x = freqs[idx3.x] * input[idx3.y];
    glm::vec3 s, c;
    /*__sincosf(x.x, &s.x, &c.x);
    __sincosf(x.y, &s.y, &c.y);
    __sincosf(x.z, &s.z, &c.z);
    o_encoded[offset + idx3.x * 2 + 1] = s;
    o_encoded[offset + idx3.x * 2 + 2] = c;*/
    o_encoded[offset + idx3.x * 2 + 1] = glm::sin(x);
    o_encoded[offset + idx3.x * 2 + 2] = glm::cos(x);
}

void Encoder::encode(sptr<CudaArray<float>> o_encoded, sptr<CudaArray<glm::vec3>> input)
{
    dim3 blockSize(_multires, 1024 / _multires);
    dim3 gridSize(1, (uint)ceil(input->n() / (float)blockSize.y));
    cu_encode<<<gridSize, blockSize>>>((glm::vec3 *)o_encoded->getBuffer(),
                                       *input, *_freqs, input->n());
    CHECK_EX(cudaGetLastError());
}

void Encoder::_genFreqArray()
{
    float *arr = new float[_multires];
    arr[0] = 1.0f;
    for (auto i = 1; i < _multires; ++i)
        arr[i] = arr[i - 1] * 2.0f;
    _freqs = sptr<CudaArray<float>>(new CudaArray<float>(_multires));
    cudaMemcpy(_freqs->getBuffer(), arr, _multires * sizeof(float), cudaMemcpyHostToDevice);
    delete[] arr;
}
