#include "Sampler.h"
#include "thread_index.h"

__device__ glm::vec3 _raySphereIntersect(glm::vec3 p, glm::vec3 v, float r, float &o_depth)
{
    float pp = glm::dot(p, p);
    float vv = glm::dot(v, v);
    float pv = glm::dot(p, v);
    o_depth = (sqrtf(pv * pv - vv * (pp - r * r)) - pv) / vv;
    return p + o_depth * v;
}

__device__ float _getAngle(float x, float y)
{
    return -atan(x / y) + (y < 0) * (float)M_PI + 0.5f * (float)M_PI;
}

/**
 * Dispatch with block_size=(n_samples, 1024)
 */
__global__ void cu_sampleOnRays(glm::vec3 *o_sphericalCoords, float *o_depths, glm::vec3 *rays,
                                glm::vec3 rayCenter, float range0, float rangeStep, uint n)
{
    glm::uvec3 idx3 = IDX3;
    uint rayIdx = flattenIdx({idx3.y, idx3.z, 0});
    if (rayIdx >= n)
        return;
    uint idx = flattenIdx(idx3);
    float r_reciprocal = rangeStep * idx3.x + range0;
    glm::vec3 p = _raySphereIntersect(rayCenter, rays[rayIdx], 1.0f / r_reciprocal, o_depths[idx]);
    o_sphericalCoords[idx] = glm::vec3(r_reciprocal, _getAngle(p.x, p.z), acos(p.y * r_reciprocal));
}

void Sampler::sampleOnRays(sptr<CudaArray<glm::vec3>> o_sphericalCoords, sptr<CudaArray<float>> o_depths,
                           sptr<CudaArray<glm::vec3>> rays,
                           glm::vec3 rayCenter)
{
    dim3 blockSize(_samples, 1024 / _samples);
    dim3 gridSize(1, (uint)ceil(rays->n() / (float)blockSize.y));
    cu_sampleOnRays<<<gridSize, blockSize>>>(*o_sphericalCoords, *o_depths, *rays, rayCenter,
                                             _dispRange.x,
                                             (_dispRange.y - _dispRange.x) / (_samples - 1),
                                             rays->n());
    CHECK_EX(cudaGetLastError());
}