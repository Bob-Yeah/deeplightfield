#pragma once
#include <map>
#include <vector>

class Resource
{
public:
	virtual ~Resource() {}

	virtual void *getBuffer() const = 0;

	virtual size_t size() const = 0;
};

class CudaBuffer : public Resource
{
public:
	CudaBuffer(void *buffer = nullptr, size_t size = 0) : _buffer(buffer), _ownBuffer(false), _size(size) {}
	CudaBuffer(size_t size) : _buffer(nullptr), _ownBuffer(true), _size(size)
	{
		CHECK_EX(cudaMalloc(&_buffer, size));
	}
	CudaBuffer(const CudaBuffer &rhs) = delete;

	virtual ~CudaBuffer()
	{
		if (!_ownBuffer || _buffer == nullptr)
			return;
		try
		{
			CHECK_EX(cudaFree(_buffer));
		}
		catch (std::exception &ex)
		{
			Logger::instance.warning(std::string("Exception raised in destructor: ") + ex.what());
		}
		_buffer = nullptr;
		_ownBuffer = false;
	}

	virtual void *getBuffer() const { return _buffer; }

	virtual size_t size() const { return _size; }

private:
	void *_buffer;
	bool _ownBuffer;
	size_t _size;
};

template <typename T>
class CudaArray : public CudaBuffer
{
public:
	CudaArray(size_t n) : CudaBuffer(n * sizeof(T)) {}
	CudaArray(T *buffer, size_t n) : CudaBuffer(buffer, n * sizeof(T)) {}
	CudaArray(const CudaArray<T> &rhs) = delete;

	size_t n() const { return size() / sizeof(T); }

	operator T *() { return (T *)getBuffer(); }
};

class GraphicsResource : public Resource
{
public:
	cudaGraphicsResource_t getHandler() { return _res; }

	virtual ~GraphicsResource()
	{
		if (_res == nullptr)
			return;
		try
		{
			CHECK_EX(cudaGraphicsUnregisterResource(_res));
		}
		catch (std::exception &ex)
		{
			Logger::instance.warning(std::string("Exception raised in destructor: ") + ex.what());
		}
		_res = nullptr;
	}

	virtual size_t size() const { return _size; }

protected:
	cudaGraphicsResource_t _res;
	size_t _size;

	GraphicsResource() : _res(nullptr), _size(0) {}
};

template<typename T>
class GlTextureResource : public GraphicsResource
{
public:
	GlTextureResource(GLuint textureID, glm::uvec2 textureSize)
	{
		CHECK_EX(cudaGraphicsGLRegisterImage(&_res, textureID, GL_TEXTURE_2D,
											 cudaGraphicsRegisterFlagsWriteDiscard));
		_size = textureSize.x * textureSize.y * sizeof(T);
		_textureSize = textureSize;
	}

	virtual ~GlTextureResource()
	{
		cudaGraphicsUnmapResources(1, &_res, 0);
	}

	virtual void *getBuffer() const {
		cudaArray_t buffer;
		try{
			CHECK_EX(cudaGraphicsSubResourceGetMappedArray(&buffer, _res, 0, 0));
		} catch (...) {
			return nullptr;
		}
		return buffer;
	}

	operator T *() { return (T *)getBuffer(); }
	
	glm::uvec2 textureSize () { return _textureSize; }

private:
	glm::uvec2 _textureSize;

};

class Resources
{
public:
	std::map<std::string, Resource *> resources;
	std::vector<cudaGraphicsResource_t> graphicsResources;

	void addResource(const std::string &name, Resource *res)
	{
		auto gres = dynamic_cast<GraphicsResource *>(res);
		if (gres != nullptr)
			graphicsResources.push_back(gres->getHandler());
		resources[name] = res;
	}

	void clear()
	{
		resources.clear();
		graphicsResources.clear();
	}
};

template <typename T>
void dumpFloatArray(std::ostream &so, CudaArray<T> &arr, size_t maxDumpRows = 0,
					size_t elemsPerRow = 1)
{
	T *hostArr = new T[arr.n()];
	cudaMemcpy(hostArr, arr.getBuffer(), arr.n() * sizeof(T), cudaMemcpyDeviceToHost);
	dumpHostBuffer<float>(so, hostArr, arr.n() * sizeof(T),
						  sizeof(T) / sizeof(float) * elemsPerRow, maxDumpRows);
	delete[] hostArr;
}