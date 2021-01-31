#include "Enhancement.h"
#include "thread_index.h"

#define max(__a__, __b__) (__a__ > __b__ ? __a__ : __b__)
#define min(__a__, __b__) (__a__ < __b__ ? __a__ : __b__)

__global__ void cu_boxFilter(glm::vec4 *o_filtered, glm::vec4 *imageData,
                             glm::uvec2 res)
{
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    glm::vec4 c;
    float n = 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            glm::ivec2 idx2_ = (glm::ivec2)idx2 + glm::ivec2(dx, dy);
            if (idx2_.x < 0 || idx2_.x >= res.x || idx2_.y < 0 || idx2_.y >= res.y)
                continue;
            int idx_ = idx2_.x + idx2_.y * res.x;
            c += imageData[idx_];
            n += 1.0f;
        }
    }
    o_filtered[idx] = c / n;
}

__global__ void cu_constrastEnhance(glm::vec4 *io_imageData,
                                    glm::vec4 *filtered, float cScale, glm::uvec2 res)
{
    glm::uvec2 idx2 = IDX2;
    if (idx2.x >= res.x || idx2.y >= res.y)
        return;
    uint idx = idx2.x + idx2.y * res.x;
    glm::vec4 c = filtered[idx] + (io_imageData[idx] - filtered[idx]) * cScale;
    io_imageData[idx].r = min(max(c.r, 0.0f), 1.0f);
    io_imageData[idx].g = min(max(c.g, 0.0f), 1.0f);
    io_imageData[idx].b = min(max(c.b, 0.0f), 1.0f);
}

Enhancement::Enhancement(
    glm::uvec2 res) : _res(res),
                      _boxFiltered(new CudaArray<glm::vec4>(res.x * res.y))
{
}

void Enhancement::run(sptr<CudaArray<glm::vec4>> imageData, float sigma, float fe)
{
    dim3 blockSize(32, 32);
    dim3 gridSize((uint)ceil(_res.x / (float)blockSize.x),
                  (uint)ceil(_res.y / (float)blockSize.y));
    cu_boxFilter<<<gridSize, blockSize>>>(*_boxFiltered, *imageData, _res);
    cu_constrastEnhance<<<gridSize, blockSize>>>(*imageData, *_boxFiltered,
                                                 1.0f + sigma * fe, _res);
}