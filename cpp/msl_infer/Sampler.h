#pragma once
#include "Common.h"

class Sampler
{
public:
    Sampler(glm::vec2 depthRange, uint samples) : _dispRange(1.0f / depthRange.x, 1.0f / depthRange.y),
                                                  _samples(samples) {}

    void sampleOnRays(sptr<CudaArray<glm::vec3>> o_sphericalCoords,
                      sptr<CudaArray<float>> o_depths,
                      sptr<CudaArray<glm::vec3>> rays,
                      glm::vec3 rayCenter);

private:
    glm::vec2 _dispRange;
    uint _samples;
};