#include "Nmsl2.h"
#include <time.h>

Nmsl2::Nmsl2(int batchSize, int samples) : Msl(batchSize, samples),
										   resRaw1(nullptr), resRaw2(nullptr),
										   fcNet1(nullptr), fcNet2(nullptr), catNet(nullptr) {}

bool Nmsl2::load(const std::string &netDir)
{
	fcNet1 = new Net();
	fcNet2 = new Net();
	catNet = new Net();
	if (!fcNet1->load(netDir + "fc1.trt") || !fcNet2->load(netDir + "fc2.trt") ||
		!catNet->load(netDir + "cat.trt"))
		return false;
	resRaw1 = sptr<Resource>(new CudaBuffer(batchSize * samples / 2 * sizeof(float4)));
	resRaw2 = sptr<Resource>(new CudaBuffer(batchSize * samples / 2 * sizeof(float4)));
	return true;
}

void Nmsl2::bindResources(Resource *resEncoded, Resource *resDepths, Resource *resColors)
{
	fcNet1->bindResource("Encoded", resEncoded);
	fcNet1->bindResource("Raw", resRaw1.get());
	fcNet2->bindResource("Encoded", resEncoded);
	fcNet2->bindResource("Raw", resRaw2.get());
	catNet->bindResource("Raw1", resRaw1.get());
	catNet->bindResource("Raw2", resRaw2.get());
	catNet->bindResource("Depths", resDepths);
	catNet->bindResource("Colors", resColors);
}

bool Nmsl2::infer()
{
	//CudaStream stream1, stream2;
	if (!fcNet1->infer())
		return false;
	if (!fcNet2->infer())
		return false;
	if (!catNet->infer())
		return false;
	return true;
}

bool Nmsl2::dispose()
{
	if (fcNet1 != nullptr)
	{
		fcNet1->dispose();
		delete fcNet1;
		fcNet1 = nullptr;
	}
	if (fcNet2 != nullptr)
	{
		fcNet2->dispose();
		delete fcNet2;
		fcNet2 = nullptr;
	}
	if (catNet != nullptr)
	{
		catNet->dispose();
		delete catNet;
		catNet = nullptr;
	}
	resRaw1 = nullptr;
	resRaw2 = nullptr;
	return true;
}
