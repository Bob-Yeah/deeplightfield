#pragma once
#include "Common.h"

class Renderer {
public:
    Renderer();

    void render(sptr<CudaArray<glm::vec4>> o_colors, sptr<CudaArray<glm::vec4>> layeredColors);

};