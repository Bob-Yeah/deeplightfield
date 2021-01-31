#pragma once
#include "Msl.h"

class Nmsl2 : public Msl
{
public:
	sptr<Resource> resRaw1;
	sptr<Resource> resRaw2;
	Net *fcNet1;
	Net *fcNet2;
	Net *catNet;

	Nmsl2(int batchSize, int samples);

	virtual bool load(const std::string &netDir);

	virtual void bindResources(Resource *resEncoded, Resource *resDepths, Resource *resColors);

	virtual bool infer();

	virtual bool dispose();

};
