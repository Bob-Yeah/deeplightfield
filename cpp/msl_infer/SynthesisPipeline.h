#pragma once
#include "Common.h"
#include "InferPipeline.h"
#include "View.h"
#include "Enhancement.h"

class SynthesisPipeline
{
public:
    SynthesisPipeline(const std::string &netDir, bool isNmsl,
                      uint batchSize, uint samples);

    void run(View& view);

    GLuint getGlResultTexture(int index);

protected:
    uint _batchSize;
    uint _samples;
    std::vector<GLuint> _glResultTextures;
    GLuint _glResultBuffer;
    sptr<InferPipeline> _inferPipeline;
    sptr<CudaArray<glm::vec3>> _rays;
    sptr<CudaArray<glm::vec4>> _colors;

    virtual void _genRays(View& view) = 0;
    virtual void _enhance() = 0;
    virtual void _uploadResultToTextures() = 0;

    GLuint _createGlResultTexture(glm::uvec2 res);
    GLuint _createGlResultBuffer(uint elements);

};

class FoveaSynthesisPipeline : public SynthesisPipeline
{
public:
    FoveaSynthesisPipeline(glm::uvec2 res, float fov, uint samples);

protected:
    virtual void _genRays(View& view);
    virtual void _enhance();
    virtual void _uploadResultToTextures();

private:
    Camera _foveaCamera;
    sptr<Enhancement> _enhancement;
};

class PeriphSynthesisPipeline : public SynthesisPipeline
{
public:
    PeriphSynthesisPipeline(glm::uvec2 midRes, float midFov,
                            glm::uvec2 periphRes, float periphFov,
                            uint samples);

protected:
    virtual void _genRays(View& view);
    virtual void _enhance();
    virtual void _uploadResultToTextures();

private:
    Camera _midCamera;
    Camera _periphCamera;
    sptr<CudaArray<glm::vec3>> _midRays;
    sptr<CudaArray<glm::vec3>> _periphRays;
    sptr<CudaArray<glm::vec4>> _midColors;
    sptr<CudaArray<glm::vec4>> _periphColors;
    sptr<Enhancement> _midEnhancement;
    sptr<Enhancement> _periphEnhancement;
};