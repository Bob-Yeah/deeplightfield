#pragma once
#include "Common.h"

class Enhancement
{
public:
    Enhancement(glm::uvec2 res);

    void run(sptr<CudaArray<glm::vec4>> imageData, float sigma, float fe);

private:
    glm::uvec2 _res;
    sptr<CudaArray<glm::vec4>> _boxFiltered;

};