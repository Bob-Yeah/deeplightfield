#pragma once
#include "Common.h"
#include "Net.h"

class Msl
{
public:
	int batchSize;
	int samples;
    Net *net;

	Msl(int batchSize, int samples);

	virtual bool load(const std::string &netDir);

	virtual void bindResources(Resource *resEncoded, Resource *resDepths, Resource *resColors);

	virtual bool infer();

	virtual bool dispose();
};
