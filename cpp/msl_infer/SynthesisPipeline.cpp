#include "SynthesisPipeline.h"

SynthesisPipeline::SynthesisPipeline(
    const std::string &netDir, bool isNmsl, uint batchSize,
    uint samples) : _batchSize(batchSize),
                    _samples(samples),
                    _inferPipeline(new InferPipeline(netDir, isNmsl, batchSize, samples)),
                    _rays(new CudaArray<glm::vec3>(batchSize)),
                    _colors(new CudaArray<glm::vec4>(batchSize))
{
    _glResultBuffer = _createGlResultBuffer(_batchSize);
}

void SynthesisPipeline::run(View &view)
{
    CudaEvent eStart, eGenRays, eInferred, eEnhanced;

    cudaEventRecord(eStart);

    _genRays(view);

    cudaEventRecord(eGenRays);

    _inferPipeline->run(_colors, _rays, view.t(), true);

    cudaEventRecord(eInferred);

    _enhance();

    cudaEventRecord(eEnhanced);

    CHECK_EX(cudaDeviceSynchronize());

    float timeTotal, timeGenRays, timeInfer, timeEnhance;
    cudaEventElapsedTime(&timeTotal, eStart, eEnhanced);
    cudaEventElapsedTime(&timeGenRays, eStart, eGenRays);
    cudaEventElapsedTime(&timeInfer, eGenRays, eInferred);
    cudaEventElapsedTime(&timeEnhance, eInferred, eEnhanced);
    {
        std::ostringstream sout;
        sout << typeid(*this).name() << " => Total: " << timeTotal << "ms (Gen rays: " << timeGenRays
             << "ms, Infer: " << timeInfer << "ms, Enhance: " << timeEnhance << "ms)";
        Logger::instance.info(sout.str());
    }

    // Copy result from Cuda array to OpenGL buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _glResultBuffer);
    void *bufferData = glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    cudaMemcpy(bufferData, _colors->getBuffer(), _colors->size(), cudaMemcpyDeviceToHost);
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    _uploadResultToTextures();
}

GLuint SynthesisPipeline::getGlResultTexture(int index)
{
    return _glResultTextures[index];
}

GLuint SynthesisPipeline::_createGlResultTexture(glm::uvec2 res)
{
    GLuint textureID;
    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, res.x, res.y, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);
    glDisable(GL_TEXTURE_2D);
    return textureID;
}

GLuint SynthesisPipeline::_createGlResultBuffer(uint elements)
{
    GLuint glBuffer;
    glGenBuffers(1, &glBuffer);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, glBuffer);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, elements * sizeof(glm::vec4),
                 nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    return glBuffer;
}

FoveaSynthesisPipeline::FoveaSynthesisPipeline(
    glm::uvec2 res, float fov,
    uint samples) : SynthesisPipeline("../nets/fovea_mono/", false, res.x * res.y, samples),
                    _foveaCamera(fov, res / 2u, res),
                    _enhancement(new Enhancement(res))
{
    _glResultTextures.push_back(_createGlResultTexture(res));
}

void FoveaSynthesisPipeline::_genRays(View &view)
{
    view.transVectors(_rays, _foveaCamera.localRays());
}

void FoveaSynthesisPipeline::_enhance()
{
    _enhancement->run(_colors, 3.0f, 0.2f);
}

void FoveaSynthesisPipeline::_uploadResultToTextures()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _glResultBuffer);
    glBindTexture(GL_TEXTURE_2D, _glResultTextures[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _foveaCamera.res().x, _foveaCamera.res().y,
                    GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

PeriphSynthesisPipeline::PeriphSynthesisPipeline(
    glm::uvec2 midRes, float midFov, glm::uvec2 periphRes, float periphFov,
    uint samples) : SynthesisPipeline("../nets/periph/", false,
                                      midRes.x * midRes.y + periphRes.x * periphRes.y,
                                      samples),
                    _midCamera(midFov, midRes / 2u, midRes),
                    _periphCamera(periphFov, periphRes / 2u, periphRes),
                    _midEnhancement(new Enhancement(midRes)),
                    _periphEnhancement(new Enhancement(periphRes))
{
    uint midPixels = midRes.x * midRes.y;
    uint periphPixels = periphRes.x * periphRes.y;
    _midRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(*_rays, midPixels));
    _periphRays = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(
        (glm::vec3 *)*_rays + midPixels, periphPixels));
    _glResultTextures.push_back(_createGlResultTexture(midRes));
    _glResultTextures.push_back(_createGlResultTexture(periphRes));
    _midColors = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(*_colors, midPixels));
    _periphColors = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(
        (glm::vec4 *)*_colors + midPixels, periphPixels));
}

void PeriphSynthesisPipeline::_genRays(View &view)
{
    view.transVectors(_midRays, _midCamera.localRays());
    view.transVectors(_periphRays, _periphCamera.localRays());
}

void PeriphSynthesisPipeline::_enhance()
{
    _midEnhancement->run(_midColors, 5.0f, 0.2f);
    _periphEnhancement->run(_periphColors, 5.0f, 0.2f);
}


void PeriphSynthesisPipeline::_uploadResultToTextures()
{
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _glResultBuffer);
    glBindTexture(GL_TEXTURE_2D, _glResultTextures[0]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _midCamera.res().x, _midCamera.res().y,
                    GL_RGBA, GL_FLOAT, 0);
    glBindTexture(GL_TEXTURE_2D, _glResultTextures[1]);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                    _periphCamera.res().x, _periphCamera.res().y, GL_RGBA, GL_FLOAT,
                    (void *)(_midCamera.res().x * _midCamera.res().y * sizeof(glm::vec4)));
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}
