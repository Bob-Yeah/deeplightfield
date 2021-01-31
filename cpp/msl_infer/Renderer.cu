#include "Renderer.h"
#include "thread_index.h"

/// Dispatch (n, 1, 1)
__global__ void cu_render(glm::vec4 *o_colors, glm::vec4 *layeredColors, uint samples, uint n)
{
    glm::uvec3 idx3 = IDX3;
    if (idx3.x >= n)
        return;
    glm::vec4 outColor;
    for (int i = samples - 1; i >= 0; --i)
    {
        glm::vec4 c = layeredColors[idx3.x * samples + i];
        outColor = outColor * (1 - c.a) + c * c.a;
    }
    outColor.a = 1.0f;
    o_colors[idx3.x] = outColor;
}

Renderer::Renderer()
{
}

void Renderer::render(sptr<CudaArray<glm::vec4>> o_colors,
                      sptr<CudaArray<glm::vec4>> layeredColors)
{
    dim3 blockSize(1024);
    dim3 gridSize((uint)ceil(o_colors->n() / (float)blockSize.x));
    cu_render<<<gridSize, blockSize>>>(*o_colors, *layeredColors, layeredColors->n() / o_colors->n(),
                                       o_colors->n());
    CHECK_EX(cudaGetLastError());
}