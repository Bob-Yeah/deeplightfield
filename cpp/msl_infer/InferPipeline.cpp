#include "InferPipeline.h"
#include "Nmsl2.h"

InferPipeline::InferPipeline(
    const std::string &netDir, bool isNmsl, uint batchSize,
    uint samples) : _batchSize(batchSize),
                    _samples(samples),
                    _sampler(new Sampler({1.0f, 50.0f}, samples)),
                    _encoder(new Encoder(10)),
                    _renderer(new Renderer()),
                    _net(isNmsl ? new Nmsl2(batchSize, samples) : new Msl(batchSize, samples))
{
    uint batchSizeForNet = _batchSize * _samples;
    _sphericalCoords = sptr<CudaArray<glm::vec3>>(new CudaArray<glm::vec3>(batchSizeForNet));
    _depths = sptr<CudaArray<float>>(new CudaArray<float>(batchSizeForNet));
    _encoded = sptr<CudaArray<float>>(new CudaArray<float>(batchSizeForNet * _encoder->outDim()));
    _layeredColors = sptr<CudaArray<glm::vec4>>(new CudaArray<glm::vec4>(batchSizeForNet));
    _net->load(netDir);
    _net->bindResources(_encoded.get(), _depths.get(), _layeredColors.get());
}

void InferPipeline::run(sptr<CudaArray<glm::vec4>> o_colors,
                        sptr<CudaArray<glm::vec3>> rays,
                        glm::vec3 rayOrigin, bool showPerf)
{

    CudaEvent eStart, eSampled, eEncoded, eInferred, eRendered;

    cudaEventRecord(eStart);

    _sampler->sampleOnRays(_sphericalCoords, _depths, rays, rayOrigin);

    cudaEventRecord(eSampled);

    _encoder->encode(_encoded, _sphericalCoords);

    cudaEventRecord(eEncoded);

    _net->infer();

    cudaEventRecord(eInferred);

    _renderer->render(o_colors, _layeredColors);

    cudaEventRecord(eRendered);

    if (showPerf) {
        CHECK_EX(cudaDeviceSynchronize());

        float timeTotal, timeSample, timeEncode, timeInfer, timeRender;
        cudaEventElapsedTime(&timeTotal, eStart, eRendered);
        cudaEventElapsedTime(&timeSample, eStart, eSampled);
        cudaEventElapsedTime(&timeEncode, eSampled, eEncoded);
        cudaEventElapsedTime(&timeInfer, eEncoded, eInferred);
        cudaEventElapsedTime(&timeRender, eInferred, eRendered);

        std::ostringstream sout;
        sout << "Infer pipeline: " << timeTotal << "ms (Sample: " << timeSample
             << "ms, Encode: " << timeEncode << "ms, Infer: "
             << timeInfer << "ms, Render: " << timeRender << "ms)";
        Logger::instance.info(sout.str());
    }
	/*
	{
		std::ostringstream sout;
		sout << "Rays:" << std::endl;
		dumpFloatArray(sout, *rays, 10);
		Logger::instance.info(sout.str());
	}
	{
		std::ostringstream sout;
		sout << "Spherical coords:" << std::endl;
		dumpFloatArray(sout, *sphericalCoords, 10);
		Logger::instance.info(sout.str());
	}
	{
		std::ostringstream sout;
		sout << "Depths:" << std::endl;
		dumpFloatArray(sout, *depths, 10);
		Logger::instance.info(sout.str());
	}
	{
		std::ostringstream sout;
		sout << "Encoded:" << std::endl;
		dumpFloatArray(sout, *encoded, 10, encoder.outDim());
		Logger::instance.info(sout.str());
	}
	*/
}