#pragma once
#include "Common.h"
#include "../msl_infer/Sampler.h"
#include "../msl_infer/Encoder.h"
#include "../msl_infer/Renderer.h"
#include "../msl_infer/Msl.h"

class InferPipeline
{
public:
    InferPipeline(const std::string &netDir, bool isNmsl, uint batchSize, uint samples);

    void run(sptr<CudaArray<glm::vec4>> o_colors, sptr<CudaArray<glm::vec3>> rays,
             glm::vec3 rayOrigin, bool showPerf = false);

private:
    uint _batchSize;
    uint _samples;
    sptr<Sampler> _sampler;
    sptr<Encoder> _encoder;
    sptr<Renderer> _renderer;
    sptr<Msl> _net;
    sptr<CudaArray<glm::vec3>> _sphericalCoords;
    sptr<CudaArray<float>> _depths;
    sptr<CudaArray<float>> _encoded;
    sptr<CudaArray<glm::vec4>> _layeredColors;
};