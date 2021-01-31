#pragma once
#include "Common.h"

class Encoder {
public:
    Encoder(uint multires) : _multires(multires) { _genFreqArray(); }

    uint outDim() const { return _multires * 6 + 3; }
    void encode(sptr<CudaArray<float>> o_encoded, sptr<CudaArray<glm::vec3>> input);

private:
    uint _multires;
    sptr<CudaArray<float>> _freqs;

    void _genFreqArray();

};