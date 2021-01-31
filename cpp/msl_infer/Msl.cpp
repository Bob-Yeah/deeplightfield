#include "Msl.h"
#include <time.h>

Msl::Msl(int batchSize, int samples) : batchSize(batchSize), samples(samples), net(nullptr) {}

bool Msl::load(const std::string &netDir)
{
    net = new Net();
    if (!net->load(netDir + "msl.trt"))
        return false;
    return true;
}

void Msl::bindResources(Resource *resEncoded, Resource *resDepths, Resource *resColors)
{
    net->bindResource("Encoded", resEncoded);
    net->bindResource("Depths", resDepths);
    net->bindResource("Colors", resColors);
}

bool Msl::infer()
{
    if (!net->infer())
        return false;
    return true;
}

bool Msl::dispose()
{
    if (net != nullptr)
    {
        net->dispose();
        delete net;
        net = nullptr;
    }
    return true;
}
