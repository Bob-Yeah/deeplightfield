#pragma once
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>
#include <sstream>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include "../glm/glm.hpp"
#include "Logger.h"

inline unsigned int getElementSize(nv::DataType t)
{
	switch (t)
	{
	case nv::DataType::kINT32:
		return 4;
	case nv::DataType::kFLOAT:
		return 4;
	case nv::DataType::kHALF:
		return 2;
	case nv::DataType::kBOOL:
	case nv::DataType::kINT8:
		return 1;
	}
	throw std::runtime_error("Invalid DataType.");
	return 0;
}

template <typename T>
void dumpRow(std::ostream &os, T* buf, size_t n)
{
	os << buf[0];
	for (size_t i = 1; i < n; ++i) {
		os << " " << buf[i];
	}
	os << std::endl;
}

template <typename T>
void dumpHostBuffer(std::ostream &os, void *buf, size_t bufSize, size_t rowCount, size_t maxDumpRows = 0)
{
	T *typedBuf = static_cast<T *>(buf);
	size_t numItems = bufSize / sizeof(T);
	size_t nInLastRow = numItems % rowCount;
	size_t rows;
	if (nInLastRow == 0) {
		rows = numItems / rowCount;
		nInLastRow = rowCount;
	} else {
		rows = numItems / rowCount + 1;
	}
	if (maxDumpRows == 0) {
		for (size_t i = 0; i < rows - 1; ++i) {
			dumpRow(os, typedBuf, rowCount);
			typedBuf += rowCount;
		}
		dumpRow(os, typedBuf, nInLastRow);
	} else {
		for (size_t i = 0; i < maxDumpRows / 2; ++i)
			dumpRow(os, typedBuf + i * rowCount, rowCount);
		os << "..." << std::endl;
		for (size_t i = rows - maxDumpRows + maxDumpRows / 2; i < rows - 1; ++i)
			dumpRow(os, typedBuf + i * rowCount, rowCount);
		dumpRow(os, typedBuf + (rows - 1) * rowCount, nInLastRow);
	}
}

class CudaStream
{
public:
	CudaStream()
	{
		cudaStreamCreate(&stream);
	}

	operator cudaStream_t()
	{
		return stream;
	}

	virtual ~CudaStream()
	{
		cudaStreamDestroy(stream);
	}

private:
	cudaStream_t stream;
};

class CudaEvent
{
public:
	CudaEvent()
	{
		cudaEventCreate(&mEvent);
	}

	operator cudaEvent_t()
	{
		return mEvent;
	}

	virtual ~CudaEvent()
	{
		cudaEventDestroy(mEvent);
	}

private:
	cudaEvent_t mEvent;
};

struct CudaMapScope
{
	std::vector<cudaGraphicsResource_t> resources_;
	cudaStream_t stream_;

	CudaMapScope(const std::vector<cudaGraphicsResource_t> &resources,
				 cudaStream_t stream = nullptr) : resources_(resources), stream_(stream) {}

	~CudaMapScope()
	{
		if (!resources_.empty())
			cudaGraphicsUnmapResources(resources_.size(),
									   resources_.data(), stream_);
	}

	cudaError_t map()
	{
		if (!resources_.empty())
			return cudaGraphicsMapResources(resources_.size(),
											resources_.data(), stream_);
		return cudaSuccess;
	}
};

template <typename T>
struct Destroy
{
	void operator()(T *t)
	{
		if (t != nullptr)
			t->destroy();
	}
};

template <class T>
using uptr = std::unique_ptr<T, ::Destroy<T>>;
template <class T>
using sptr = std::shared_ptr<T>;

#define INTERVAL(__start__, __end__) (((__end__) - (__start__)) / (float)CLOCKS_PER_SEC * 1000)

#include "Resource.h"
#include "Formatter.h"