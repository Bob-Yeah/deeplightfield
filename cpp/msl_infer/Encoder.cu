#include "Encoder.h"
#include "thread_index.h"

/// idx3.z = 0: x, y, z, sin(x), sin(y), sin(z), cos(x), cos(y), cos(z)
/// idx3.z = 1: sin(2x), sin(2y), sin(2z), cos(2x), cos(2y), cos(2z)
/// ...
/// idx3.z = n_freq-1: sin(2^(n_freq-1)x), sin(2^(n_freq-1)y), sin(2^(n_freq-1)z),
///                    cos(2^(n_freq-1)x), cos(2^(n_freq-1)y), cos(2^(n_freq-1)z)
/// Dispatch (n_batch, n_chns, n_freqs)
__global__ void cu_encode(float *output, float *input, float *freqs, uint n)
{
    glm::uvec3 idx3 = IDX3;
    if (idx3.x >= n)
        return;
    uint n = blockDim.x, inChns = blockDim.y, nFreqs = blockDim.z;
    uint i = idx3.x, chn = idx3.y, freq = idx3.z;
    uint elem = i * inChns + chn;
    uint outChns = inChns * (nFreqs * 2 + 1);
    uint base = i * outChns + chn;
    if (idx3.x == 0)
        output[base] = input[elem];
    float x = freqs[freq] * input[elem];
    float s, c;
    __sincosf(x, &s, &c);
    output[base + inChns * (freq * 2 + 1)] = s;
    output[base + inChns * (freq * 2 + 2)] = c;
}

void Encoder::encode(sptr<CudaArray<float>> output, sptr<CudaArray<float>> input)
{
    dim3 blkSize(1024 / _chns / _multires, _chns, _multires);
    dim3 grdSize((uint)ceil(input->n() / (float)blkSize.x), 1, 1);
    cu_encode<<<grdSize, blkSize>>>(output->getBuffer(), *input, *_freqs, input->n());
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
