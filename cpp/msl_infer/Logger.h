#pragma once
#include <iostream>
#include <string>
#include <sstream>
#include <cuda_runtime_api.h>
#include <NvInfer.h>

namespace nv = nvinfer1;


typedef void(*ExternalLogFuncPtr)(int severity, const char*);


class Logger : public nv::ILogger {
public:
	ExternalLogFuncPtr externalLogFunc = nullptr;
	int logLevel = 1;
	static Logger instance;

	void info(const std::string& msg) {
		log(nv::ILogger::Severity::kINFO, msg.c_str());
	}

	void warning(const std::string& msg) {
		log(nv::ILogger::Severity::kWARNING, msg.c_str());
	}

	void error(const std::string& msg) {
		log(nv::ILogger::Severity::kERROR, msg.c_str());
	}

	bool checkErr(cudaError_t err, const char* file, int line) {
		if (err == cudaSuccess)
			return true;
		std::ostringstream sout;
		sout << "Cuda error " << cudaGetErrorName(err) << " at "
			<< file << " (Line " << line << "): " << cudaGetErrorString(err);
		error(sout.str());
		return false;
	}

	virtual void log(nv::ILogger::Severity severity, const char* msg) override {
		if ((int)severity > logLevel)
			return;
		if (externalLogFunc == nullptr) {
			switch (severity) {
			case nv::ILogger::Severity::kVERBOSE:
				std::cout << "[VERBOSE] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kINFO:
				std::cout << "[INFO] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kWARNING:
				std::cerr << "[WARNING] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kERROR:
				std::cerr << "[ERROR] " << msg << std::endl;
				break;
			case nv::ILogger::Severity::kINTERNAL_ERROR:
				std::cerr << "[ERROR] " << msg << std::endl;
				break;
			}
			return;
		}
		externalLogFunc((int)severity, msg);
	}
};


#define CHECK(__ERR_CODE__) do { if (!Logger::instance.checkErr((__ERR_CODE__), __FILE__, __LINE__)) return false; } while (0)
#define CHECK_EX(__ERR_CODE__) do { if (!Logger::instance.checkErr((__ERR_CODE__), __FILE__, __LINE__)) throw std::exception(); } while (0)
