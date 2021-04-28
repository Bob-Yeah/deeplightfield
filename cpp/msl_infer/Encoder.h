#pragma once
#include "Common.h"

class Encoder {
public:
    Encoder(uint multires, uint chns) : _multires(multires), _chns(chns) { _genFreqArray(); }

    uint outDim() const { return _chns * (1 + _multires * 2); }
    void encode(sptr<CudaArray<float>> output, sptr<CudaArray<float>> input);

private:
    uint _multires;
    uint _chns;
    sptr<CudaArray<float>> _freqs;

    void _genFreqArray();

};