#pragma once
#include "Common.h"


class Camera {
public:
    Camera(float fov, glm::vec2 c, glm::uvec2 res);

    sptr<CudaArray<glm::vec3>> localRays();
    glm::vec2 f() const { return _f; }
    glm::vec2 c() const { return _c; }
    glm::uvec2 res() const { return _res; }

private:
    glm::vec2 _f;
    glm::vec2 _c;
    glm::uvec2 _res;
    sptr<CudaArray<glm::vec3>> _localRays;

    void _genLocalRays(bool norm);

};


class View {
public:
    View(glm::vec3 t, glm::mat3 r) : _t(t), _r(r) { }

    glm::vec3 t() const { return _t; }
    glm::mat3 r() const { return _r; }

    void transPoints(sptr<CudaArray<glm::vec3>> results,
        sptr<CudaArray<glm::vec3>> points, bool inverse = false);

    void transVectors(sptr<CudaArray<glm::vec3>> results,
        sptr<CudaArray<glm::vec3>> vectors, bool inverse = false);

private:
    glm::vec3 _t;
    glm::mat3 _r;

};