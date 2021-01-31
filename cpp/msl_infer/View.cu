#include "View.h"
#include <cuda_runtime.h>
#include "thread_index.h"

__global__ void cu_genLocalRays(glm::vec3 *o_rays, glm::vec2 f, glm::vec2 c, glm::uvec2 res)
{
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    o_rays[idx] = glm::vec3((glm::vec2(idx2) - c) / f, 1.0f);
}

__global__ void cu_genLocalRaysNormed(glm::vec3 *o_rays, glm::vec2 f, glm::vec2 c, glm::uvec2 res)
{
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    o_rays[idx] = glm::normalize(glm::vec3((glm::vec2(idx2) - c) / f, 1.0f));
}

__global__ void cu_transPoints(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::vec3 t,
                               glm::mat3 rot_t, uint n)
{
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[idx] * rot_t + t;
}

__global__ void cu_transPointsInverse(glm::vec3 *o_pts, glm::vec3 *pts, glm::vec3 t,
                                      glm::mat3 inv_rot_t, uint n)
{
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_pts[idx] = (pts[idx] - t) * inv_rot_t;
}

__global__ void cu_transVectors(glm::vec3 *o_vecs, glm::vec3 *vecs, glm::mat3 rot_t, uint n)
{
    uint idx = flattenIdx();
    if (idx >= n)
        return;
    o_vecs[idx] = vecs[idx] * rot_t;
}

Camera::Camera(float fov, glm::vec2 c, glm::uvec2 res)
{
    _f.x = _f.y = 0.5f * res.x / tan(fov * (float)M_PI / 360.0f);
    _f.y *= -1.0f;
    _c = c;
    _res = res;
}

sptr<CudaArray<glm::vec3>> Camera::localRays()
{
    if (_localRays == nullptr)
        _genLocalRays(true);
    return _localRays;
}

void Camera::_genLocalRays(bool norm)
{
    _localRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(_res.x * _res.y));
    dim3 blockSize(32, 32);
    dim3 gridSize((uint)ceil(_res.x / (float)blockSize.x),
                  (uint)ceil(_res.y / (float)blockSize.y));
    std::cout << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << std::endl;
    if (norm)
        cu_genLocalRaysNormed<<<gridSize, blockSize>>>(*_localRays, _f, _c, _res);
    else
        cu_genLocalRaysNormed<<<gridSize, blockSize>>>(*_localRays, _f, _c, _res);
}

void View::transPoints(sptr<CudaArray<glm::vec3>> results,
                       sptr<CudaArray<glm::vec3>> points, bool inverse)
{
    glm::mat3 r_t = inverse ? _r : glm::transpose(_r);
    dim3 blockSize(1024);
    dim3 gridSize((uint)ceil(points->n() / (float)blockSize.x));
    if (inverse)
        cu_transPointsInverse<<<gridSize, blockSize>>>(*results, *points, _t, r_t, points->n());
    else
        cu_transPoints<<<gridSize, blockSize>>>(*results, *points, _t, r_t, points->n());
}

void View::transVectors(sptr<CudaArray<glm::vec3>> results,
                        sptr<CudaArray<glm::vec3>> vectors, bool inverse)
{
    glm::mat3 r_t = inverse ? _r : glm::transpose(_r);
    dim3 blockSize(1024);
    dim3 gridSize((uint)ceil(vectors->n() / (float)blockSize.x));
    cu_transVectors<<<gridSize, blockSize>>>(*results, *vectors, r_t, vectors->n());
}
