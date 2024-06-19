#include <algorithm>
#include <vector>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iostream>
#include <utility>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <set>
#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


dim3 CUDA_THREADS(16, 16);
dim3 CUDA_BLOCKS(32, 32);



static void HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err)
            << " in " << std::string(file)
            << " at line " << line << "\n";
        throw std::runtime_error("cuda error");
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#undef min(a, b)
#undef max(a, b)

struct vector_t {

    friend std::istream& operator>>(std::istream& stream, vector_t& v);

    friend vector_t operator*(double, const vector_t);

    __device__ __host__
        bool operator < (vector_t const rhs) const {
        if (x < rhs.x) {
            return true;
        }
        if (x > rhs.x) {
            return false;
        }
        if (y < rhs.y) {
            return true;
        }
        if (y > rhs.y) {
            return false;
        }
        if (z < rhs.z) {
            return true;
        }
        if (z > rhs.z) {
            return false;
        }
        return false;
    }

    __device__ __host__
        bool operator !=(vector_t v) const {
        if (std::abs(x - v.x) > 1e-6) {
            return true;
        }
        if (std::abs(y - v.y) > 1e-6) {
            return true;
        }
        if (std::abs(z - v.z) > 1e-6) {
            return true;
        }
        return false;
    }

    __device__ __host__
        static const vector_t k() {
        return vector_t{ 0, 0, 1 };
    }

    __device__ __host__
        static const vector_t j() {
        return vector_t{ 0, 1, 0 };
    }

    __device__ __host__
        static const vector_t i() {
        return vector_t{ 1, 0, 0 };
    }
    double x;
    double y;
    double z;

    __device__ __host__
        double norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    __device__ __host__
        vector_t get_unit() const {
        vector_t r(*this);
        return r /= norm();
    }

    __device__ __host__
        vector_t& make_unit() {
        operator /=(norm());
        return *this;
    }

    __device__ __host__
        vector_t reflect(const vector_t& n) const {
        return  2.0 * n * (n * (*this)) - (*this);
    }

    __device__ __host__
        vector_t transform(vector_t a, vector_t b, vector_t c) {
        vector_t r;
        r.x = a.x * x + b.x * y + c.x * z;
        r.y = a.y * x + b.y * y + c.y * z;
        r.z = a.z * x + b.z * y + c.z * z;
        return r;
    }

    __device__ __host__
        double operator* (vector_t const& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __device__ __host__
        vector_t operator^(vector_t const& v) const {
        vector_t r;
        r.x = y * v.z - z * v.y;
        r.y = z * v.x - x * v.z;
        r.z = x * v.y - y * v.x;
        return r;

    }

    __device__ __host__
        vector_t operator-(const vector_t& v) const {
        vector_t r(*this);
        r.x -= v.x;
        r.y -= v.y;
        r.z -= v.z;
        return r;
    }

    __device__ __host__
        vector_t operator+(const vector_t& v) const {
        vector_t r(*this);
        r.x += v.x;
        r.y += v.y;
        r.z += v.z;
        return r;
    }

    __device__ __host__
        vector_t& operator *=(const double m) {
        x *= m;
        y *= m;
        z *= m;
        return *this;
    }

    __device__ __host__
        vector_t& operator /=(const double m) {
        x /= m;
        y /= m;
        z /= m;
        return *this;
    }

    __device__ __host__
        vector_t& operator +=(const vector_t& m) {
        x += m.x;
        y += m.y;
        z += m.z;
        return *this;
    }

    __device__ __host__
        vector_t operator /(const double m) const {
        return vector_t(*this) /= m;
    }

    __device__ __host__
        vector_t operator *(const double m) const {
        return vector_t(*this) *= m;
    }

    __device__ __host__
        vector_t operator-() const {
        return vector_t{ -x, -y, -z };
    }
};

__device__ __host__
vector_t operator*(double a, const vector_t b) {
    return vector_t{ a * b.x, a * b.y, a * b.z };
}

std::istream& operator>>(std::istream& stream, vector_t& v) {
    stream >> v.x >> v.y >> v.z;
    return stream;
}

struct color_t {
    double r;
    double g;
    double b;
    double a;
};

struct edge_t {
    vector_t a;
    vector_t b;
};


struct intersection_t {
    bool hit;
    size_t i;
    double t;
};

struct triangle_t {
    vector_t v1;
    vector_t v2;
    vector_t v3;

    __device__ __host__
    vector_t normal() const {
        return ((v2 - v1) ^ (v3 - v1)).get_unit();
    }

    std::array<edge_t, 3> exportEdges() const {
        return {
            edge_t{v1, v2},
            edge_t{v2, v3},
            edge_t{v3, v1}
        };
    }
};


struct shape_t {
    triangle_t t;
    color_t c;
    double r;
};

enum class light_kind {
    ambient,
    point,
    diode,
};

struct light_t {
    light_kind t;
    vector_t p;
    color_t c;
};



namespace input {
    struct object_props {
        vector_t origin;
        color_t color;
        double radius;
        double ref;
        int diodes_count;
        friend std::istream& operator>>(std::istream& stream, object_props& v);
    };

    std::istream& operator>>(std::istream& stream, object_props& p) {
        stream  >> p.origin
                >> p.color.r >> p.color.g >> p.color.b
                >> p.radius
                >> p.ref
                >> p.color.a
                >> p.diodes_count;
        return stream;
    }

    struct point_movement {
        double  r0, z0, phi0, ar, az, wr, wz, wphi, pr, pz;
        friend std::istream& operator>>(std::istream& stream, point_movement& v);
        vector_t getLocation(double t) const;
    };

    std::istream& operator>>(std::istream& stream, point_movement& m) {
        stream  >> m.r0 >> m.z0 >> m.phi0
                >> m.ar >> m.az
                >> m.wr >> m.wz >> m.wphi
                >> m.pr >> m.pz;
        return stream;
    }

    vector_t point_movement::getLocation(double t) const {
        double r = r0 + ar * std::sin(wr * t + pr);
        double z = z0 + az * std::sin(wz * t + pz);
        double phi = phi0 + wphi * t;
        return vector_t{ r * std::cos(phi), r * std::sin(phi), z };
    }

    struct floor {
        int w, h;
        vector_t origin;
        color_t color;
        double ref;
        friend std::istream& operator>>(std::istream& stream, floor& f);
    };

    std::istream& operator>>(std::istream& stream, floor& f) {
        stream  >> f.w >> f.h >> f.origin
                >> f.color.r >> f.color.g >> f.color.b
                >> f.ref;
        f.color.a = 0.0;
        return stream;
    };

    struct filename_template {
        std::string part1;
        std::string part2;
        friend std::istream& operator>>(std::istream& stream, filename_template& f);
        std::string getFilename(size_t frame_number) const;
    };

    std::istream& operator>>(std::istream& stream, filename_template& f) {
        std::string tmp; std::cin >> tmp;
        size_t t = tmp.find("%d");
        if (t == std::string::npos) {
            throw std::runtime_error("invalid path specifier");
        }
        f.part1 = tmp.substr(0, t);
        f.part2 = tmp.substr(t + 2);
        return stream;
    };

    std::string filename_template::getFilename(size_t frame_number) const {
        return part1 + std::to_string(frame_number) + part2;
    }
}

namespace object_presets {
    std::array<triangle_t, 36> dodecahedron() {
        const double p = (1 + std::sqrt(5)) / 2;
        const double u = 2 / (1 + std::sqrt(5));
        const double r = std::sqrt(6);
        std::array<vector_t, 20> v = {
            vector_t{-u,  0,  p} / r,
            vector_t{ u,  0,  p} / r,
            vector_t{-1,  1,  1} / r,
            vector_t{ 1,  1,  1} / r,
            vector_t{ 1, -1,  1} / r,
            vector_t{-1, -1,  1} / r,
            vector_t{ 0, -p,  u} / r,
            vector_t{ 0,  p,  u} / r,
            vector_t{-p, -u,  0} / r,
            vector_t{-p,  u,  0} / r,
            vector_t{ p,  u,  0} / r,
            vector_t{ p, -u,  0} / r,
            vector_t{ 0, -p, -u} / r,
            vector_t{ 0,  p, -u} / r,
            vector_t{ 1,  1, -1} / r,
            vector_t{ 1, -1, -1} / r,
            vector_t{-1, -1, -1} / r,
            vector_t{-1,  1, -1} / r,
            vector_t{ u,  0, -p} / r,
            vector_t{-u,  0, -p} / r
        };
        std::array<triangle_t, 36> d = {
            triangle_t{ v[7], v[3], v[13] },
            triangle_t{ v[3], v[10], v[14] },
            triangle_t{ v[0], v[1], v[3] },
            triangle_t{ v[13], v[14], v[18] },
            triangle_t{ v[11], v[15], v[18] },
            triangle_t{ v[5], v[8], v[16] },
            triangle_t{ v[12], v[16], v[19] },
            triangle_t{ v[2], v[0], v[7] },
            triangle_t{ v[0], v[3], v[7] },
            triangle_t{ v[3], v[1], v[10] },
            triangle_t{ v[17], v[13], v[19] },
            triangle_t{ v[5], v[0], v[8] },
            triangle_t{ v[8], v[0], v[9] },
            triangle_t{ v[10], v[11], v[14] },
            triangle_t{ v[2], v[13], v[17] },
            triangle_t{ v[6], v[5], v[12] },
            triangle_t{ v[0], v[2], v[9] },
            triangle_t{ v[12], v[5], v[16] },
            triangle_t{ v[4], v[12], v[15] },
            triangle_t{ v[4], v[6], v[12] },
            triangle_t{ v[11], v[4], v[15] },
            triangle_t{ v[2], v[7], v[13] },
            triangle_t{ v[1], v[4], v[11] },
            triangle_t{ v[9], v[2], v[17] },
            triangle_t{ v[13], v[3], v[14] },
            triangle_t{ v[0], v[4], v[1] },
            triangle_t{ v[14], v[11], v[18] },
            triangle_t{ v[8], v[17], v[19] },
            triangle_t{ v[15], v[12], v[18] },
            triangle_t{ v[4], v[0], v[6] },
            triangle_t{ v[16], v[8], v[19] },
            triangle_t{ v[8], v[9], v[17] },
            triangle_t{ v[0], v[5], v[6] },
            triangle_t{ v[12], v[19], v[18] },
            triangle_t{ v[10], v[1], v[11] },
            triangle_t{ v[19], v[13], v[18] },
        };
        return d;
    }

    std::array<triangle_t, 12> hexaeder() {
        const double p = 1 / (std::sqrt(12));
        std::array<vector_t, 8> v = {
            vector_t{-p,  -p,  -p}, 
            vector_t{-p,  -p,  +p}, // 1
            vector_t{-p,  +p,  -p}, 
            vector_t{-p,  +p,  +p}, // 3
            vector_t{+p,  -p,  -p},
            vector_t{+p,  -p,  +p}, // 5
            vector_t{+p,  +p,  -p},
            vector_t{+p,  +p,  +p}  // 7
        };
        std::array<triangle_t, 12> d = {
            triangle_t{ v[5], v[1], v[0] },
            triangle_t{ v[0], v[4], v[5]},
            triangle_t{ v[6], v[2], v[7] },
            triangle_t{ v[7], v[2], v[3]},

            triangle_t{ v[3], v[1], v[7]},
            triangle_t{ v[1], v[5], v[7] },

            triangle_t{ v[0], v[4], v[6] },
            triangle_t{ v[2], v[0], v[6] },
            triangle_t{ v[0], v[1], v[2] },
            triangle_t{ v[1], v[3], v[2] },
            triangle_t{ v[5], v[4], v[6] },
            triangle_t{ v[7], v[5], v[6] }
        };
        return d;
    }

    std::array<triangle_t, 8> octaedr() {
        const double p = 1 / (2 * std::sqrt(2));
        std::array<vector_t, 6> v = {
            vector_t{0,   0, +0.5},
            vector_t{0,   0, -0.5},
            vector_t{-p, -p,  0.0},
            vector_t{-p, +p,  0.0},
            vector_t{+p, -p,  0.0},
            vector_t{+p, +p,  0.0},
        };
        std::array<triangle_t, 8> d = {
            triangle_t{ v[2], v[4], v[1] },
            triangle_t{ v[4], v[5], v[1] },
            triangle_t{ v[5], v[3], v[1] },
            triangle_t{ v[3], v[2], v[1] },
            triangle_t{ v[0], v[2], v[4] },
            triangle_t{ v[0], v[4], v[5] },
            triangle_t{ v[0], v[5], v[3] },
            triangle_t{ v[0], v[3], v[2] }
        };
        return d;
    }

    std::array<triangle_t, 2> rectangle(int w, int h) {
        const double hw = w / 2.0;
        const double hh = h / 2.0;

        std::array<vector_t, 4> v = {
            vector_t{-hw, -hh, 0.0}, // a
            vector_t{-hw, +hh, 0.0}, // b
            vector_t{+hw, +hh, 0.0}, // c
            vector_t{+hw, -hh, 0.0}, // d
        };
        std::array<triangle_t, 2> d = {
            triangle_t{ v[1], v[0], v[2] },
            triangle_t{ v[2], v[0], v[3] },
        };
        return d;
    }
}

namespace antialiasing {

    struct ssaa_context_t {   // аргументы ядра
        uchar4* in, * out;     // входное и выходное изображения
        int32_t width;        // ширина сглаженного изображение
        int32_t height;       // высота сглаженного изображение
        int32_t height_ratio; // коэффициент сглаживания по высоте
        int32_t width_ratio;  // коэффициент сглаживания по ширине
    };

    __global__
        void ssaa_kernel(ssaa_context_t context) {

        const int32_t block_index = gridDim.x * blockIdx.y + blockIdx.x;        // линейный индекс блока относительно сетки
        const int32_t thread_index = blockDim.x * threadIdx.y + threadIdx.x;    // линейный индекс потока относительно блока
        const int32_t block_size = blockDim.x * blockDim.y;                     // размер блока
        const int32_t ssaa_volume = context.width_ratio * context.height_ratio; // размер блока сглаживания
        const int32_t grid_size = gridDim.x * gridDim.y * block_size;           // размер сетки в потоках
        const int32_t image_size = context.width * context.height;              // размер изображения
        const int32_t origin_width = context.width * context.width_ratio;


        for (int32_t k = block_index * block_size + thread_index;
            k < image_size; k += grid_size) {
            int32_t x = (k % context.width) * context.width_ratio;               // координаты пикселя
            int32_t y = (k / context.width) * context.height_ratio;              // координаты пикселя
            int32_t sx = 0, sy = 0, sz = 0, sw = 0;                              // суммы по значениям каналов сглаживаемого блока

            for (int32_t i = 0; i < context.width_ratio; ++i) {                  // суммируем каналы в блоке сглаживания
                for (int32_t j = 0; j < context.height_ratio; ++j) {
                    uchar4 p = context.in[(y + j) * origin_width + (x + i)];
                    sx += static_cast<unsigned int>(p.x);
                    sy += static_cast<unsigned int>(p.y);
                    sz += static_cast<unsigned int>(p.z);
                }
            }
            context.out[k].x = static_cast<unsigned char>(sx / ssaa_volume);      // усредняем эти значения и записываем
            context.out[k].y = static_cast<unsigned char>(sy / ssaa_volume);
            context.out[k].z = static_cast<unsigned char>(sz / ssaa_volume);
            context.out[k].w = 255;
        }
    }

    void ssaa_function(ssaa_context_t context) {
        const int32_t ssaa_volume = context.width_ratio * context.height_ratio;  // размер блока сглаживания
        const int32_t image_size = context.width * context.height;               // размер изображения
        const int32_t origin_width = context.width * context.width_ratio;

        for (int32_t k = 0; k < image_size; ++k) {
            int32_t x = (k % context.width) * context.width_ratio;               // координаты пикселя
            int32_t y = (k / context.width) * context.height_ratio;              // координаты пикселя
            int32_t sx = 0, sy = 0, sz = 0, sw = 0;                              // суммы по значениям каналов сглаживаемого блока

            for (int32_t i = 0; i < context.width_ratio; ++i) {                  // суммируем каналы в блоке сглаживания
                for (int32_t j = 0; j < context.height_ratio; ++j) {
                    uchar4 p = context.in[(y + j) * origin_width + (x + i)];
                    sx += static_cast<unsigned int>(p.x);
                    sy += static_cast<unsigned int>(p.y);
                    sz += static_cast<unsigned int>(p.z);
                }
            }
            context.out[k].x = static_cast<unsigned char>(sx / ssaa_volume);     // усредняем эти значения и записываем
            context.out[k].y = static_cast<unsigned char>(sy / ssaa_volume);
            context.out[k].z = static_cast<unsigned char>(sz / ssaa_volume);
            context.out[k].w = 255;
        }
    }
}

namespace info {
    std::string getOptimalConfig() {
        return
            "100                                        \n"
            "img_ % d.data                              \n"
            "1024 960 100                               \n"
            "7.0 3.0 0.0 2.0 1.0 2.0 6.0 1.0 0.0 0.0    \n"
            "2.0 0.0 0.0 0.5 0.1 1.0 4.0 1.0 0.0 0.0    \n"
            "                                           \n"
            "                                           \n"
            "0.0 - 2.5 1.0                              \n"
            "1.0 0.0 0.0                                \n"
            "2                                          \n"
            "0.9 0.4                                    \n"
            "2                                          \n"
            "                                           \n"
            "2.0 1.5 0.0                                \n"
            "0.0 1.0 0.0                                \n"
            "2                                          \n"
            "0.9 0.3                                    \n"
            "2                                          \n"
            "                                           \n"
            "                                           \n"
            "- 2.0 1.5 0.2                              \n"
            "0.0 0.7 0.7                                \n"
            "2                                          \n"
            "0.9 0.5                                    \n"
            "2                                          \n"
            "                                           \n"
            "20 20                                      \n"
            "0.0 0.0 - 1.0                              \n"
            "1 1 1                                      \n"
            "3                                          \n"
            "                                           \n"
            "                                           \n"
            "0.01 0.01 0.01                             \n"
            "                                           \n"
            "- 10 - 10 3                                \n"
            "0.4 0.4 0.4                                \n"
            "5 2                                        \n";
    }
}

namespace renders {

    __device__ __host__
        intersection_t intersect(vector_t pv, vector_t dv,
            double t_min, double t_max,
            shape_t* scene, size_t n) {
        intersection_t r;
        r.hit = false;
        for (size_t i = 0; i < n; ++i) {
            triangle_t& s = scene[i].t;
            vector_t e1 = s.v2 - s.v1;
            vector_t e2 = s.v3 - s.v1;
            vector_t p = dv ^ e2;
            double norm = p * e1;
            if (std::abs(norm) < 1e-10) {
                continue;
            }
            vector_t t = pv - s.v1;
            double u = (p * t) / norm;
            if (u < 0.0 || u > 1.0) {
                continue;
            }
            vector_t q = t ^ e1;
            double v = (q * dv) / norm;
            if (v < 0.0 || v + u > 1.0) {
                continue;
            }
            double h = (q * e2) / norm;
            if (h < t_min || h > t_max) {
                continue;
            }
            if (!r.hit || h < r.t) {
                r.hit = true;
                r.t = h;
                r.i = i;
            }
        }

        return r;

    }

    __device__ __host__
        double getTransparency(vector_t pv, vector_t dv,
            double t_min, double t_max,
            shape_t* scene, size_t n) {
        double alpha = 1.0;
        for (size_t i = 0; i < n; ++i) {
            triangle_t& s = scene[i].t;
            vector_t e1 = s.v2 - s.v1;
            vector_t e2 = s.v3 - s.v1;
            vector_t p = dv ^ e2;
            double norm = p * e1;
            if (std::abs(norm) < 1e-10) {
                continue;
            }
            vector_t t = pv - s.v1;
            double u = (p * t) / norm;
            if (u < 0.0 || u > 1.0) {
                continue;
            }
            vector_t q = t ^ e1;
            double v = (q * dv) / norm;
            if (v < 0.0 || v + u > 1.0) {
                continue;
            }
            double h = (q * e2) / norm;
            if (h < t_min || h > t_max) {
                continue;
            }
            alpha *= scene[i].c.a;
        }

        return alpha;

    }



    __host__ __device__
        color_t getLighting(vector_t pv, vector_t nv,
            vector_t vv, size_t hx,
            shape_t* scene, size_t n,
            light_t* lights, size_t m) {

        color_t x = { 0.0, 0.0, 0.0, 0.0 };
        for (size_t i = 0; i < m; ++i) {
            if (lights[i].t == light_kind::ambient) {
                x.r += lights[i].c.r;
                x.g += lights[i].c.g;
                x.b += lights[i].c.b;
            }
            else if (lights[i].t == light_kind::diode) {
                vector_t dv = lights[i].p - pv;
                double r = 1.0 / dv.norm();
                x.r += lights[i].c.r * r * r;
                x.g += lights[i].c.g * r * r;
                x.b += lights[i].c.b * r * r;
            }
            else if (lights[i].t == light_kind::point) {
                vector_t lv = lights[i].p - pv;
                double a = getTransparency(pv, lv, 1e-6, 1, scene, n);
                double nv_dot_lv = nv * lv;
                if (nv_dot_lv > 0.0) {
                    double c = a * (nv_dot_lv / (nv.norm() * lv.norm()));
                    x.r += lights[i].c.r * c;
                    x.g += lights[i].c.g * c;
                    x.b += lights[i].c.b * c;
                }
                vector_t rv = lv.reflect(nv);
                double rv_dot_vv = rv * vv;
                if (rv_dot_vv > 0.0) {
                    double c = a * (rv_dot_vv / (rv.norm() * vv.norm()));
                    c = std::pow(c, scene[hx].r);
                    x.r += lights[i].c.r * c;
                    x.g += lights[i].c.g * c;
                    x.b += lights[i].c.b * c;
                }
            }
        }
        return x;
    }


    void cpuRender(uchar4* image, size_t width, size_t height,
        shape_t* scene, size_t n,
        light_t* lights, size_t light_sources, size_t depth,
        double angle, vector_t cam, vector_t foc) {


        const double ratio = static_cast<double>(height) /
            static_cast<double>(width);
        const double dx = 2.0 / (width - 1.0);
        const double dy = 2.0 / (height - 1.0);
        const double dz = 1.0 / std::tan(angle / 2.0);

        const vector_t bz = (foc - cam).make_unit();
        const vector_t bx = (bz ^ vector_t::k()).make_unit();
        const vector_t by = (bx ^ bz).make_unit();


        const double inf = std::numeric_limits<double>::infinity();
        const double eps = 1e-6;


        for (size_t j = 0; j < height; ++j) {
            for (size_t i = 0; i < width; ++i) {
                vector_t dv{
                    (-1.0 + i * dx) * 1.0,
                    (-1.0 + j * dy) * ratio,
                    dz
                };
                dv = dv.transform(bx, by, bz);
                vector_t pv = cam;
                double r = 0, g = 0, b = 0, w = 1;
                auto ret = intersect(pv, dv, 1.0, inf, scene, n);
                for (size_t level = 0; level < depth && ret.hit; ++level) {
                    vector_t nv = scene[ret.i].t.normal();
                    pv = pv + ret.t * dv;
                    auto lt = getLighting(pv, nv, -dv, ret.i,
                        scene, n, lights,
                        light_sources);
                    const double cf = (1.0 - scene[ret.i].c.a);
                    r += scene[ret.i].c.r * lt.r * (w * cf);
                    g += scene[ret.i].c.g * lt.g * (w * cf);
                    b += scene[ret.i].c.b * lt.b * (w * cf);
                    w = std::min(1.0, w * scene[ret.i].c.a);
                    ret = intersect(pv, dv, eps, inf, scene, n);
                }
                r = std::min(255.0, 255.0 * r);
                g = std::min(255.0, 255.0 * g);
                b = std::min(255.0, 255.0 * b);
                auto& pixel = image[(height - 1 - j) * width + i];
                pixel.x = (unsigned char)(r);
                pixel.y = (unsigned char)(g);
                pixel.z = (unsigned char)(b);
                pixel.w = (unsigned char)(255);

            }
        }
    }

    __global__
    void gpuRender(uchar4* image, size_t width, size_t height,
        shape_t* scene, size_t n,
        light_t* lights, size_t light_sources, size_t depth,
        double angle, vector_t cam, vector_t foc) {


        const double ratio = static_cast<double>(height) /
            static_cast<double>(width);
        const double dx = 2.0 / (width - 1.0);
        const double dy = 2.0 / (height - 1.0);
        const double dz = 1.0 / std::tan(angle / 2.0);

        const vector_t bz = (foc - cam).make_unit();
        const vector_t bx = (bz ^ vector_t::k()).make_unit();
        const vector_t by = (bx ^ bz).make_unit();


        const double inf = 1e10;
        const double eps = 1e-6;


        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; 
                    j < height; j += blockDim.y * gridDim.y) {
            for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
                        i < width; i += blockDim.x * gridDim.x) {

                vector_t dv{
                    (-1.0 + i * dx) * 1.0,
                    (-1.0 + j * dy) * ratio,
                    dz
                };
                dv = dv.transform(bx, by, bz);
                vector_t pv = cam;
                double r = 0, g = 0, b = 0, w = 1;
                auto ret = intersect(pv, dv, 1.0, inf, scene, n);
                for (size_t level = 0; level < depth && ret.hit; ++level) {
                    vector_t nv = scene[ret.i].t.normal();
                    pv = pv + ret.t * dv;
                    auto lt = getLighting(pv, nv, -dv, ret.i,
                        scene, n, lights,
                        light_sources);
                    const double cf = (1.0 - scene[ret.i].c.a);
                    r += scene[ret.i].c.r * lt.r * (w * cf);
                    g += scene[ret.i].c.g * lt.g * (w * cf);
                    b += scene[ret.i].c.b * lt.b * (w * cf);
                    w = min(1.0, w * scene[ret.i].c.a);
                    ret = intersect(pv, dv, eps, inf, scene, n);
                }
                r = min(255.0, 255.0 * r);
                g = min(255.0, 255.0 * g);
                b = min(255.0, 255.0 * b);
                auto& pixel = image[(height - 1 - j) * width + i];
                pixel.x = (unsigned char)(r);
                pixel.y = (unsigned char)(g);
                pixel.z = (unsigned char)(b);
                pixel.w = (unsigned char)(255);

            }
        }
    }
}

namespace scene {

    template<size_t n> void transformObj(std::array<triangle_t, n>& shapes,
        vector_t origin, double radius) {               // увеличить обьект и сместить
        for (size_t i = 0; i < n; ++i) {
            shapes[i].v1 *= radius;
            shapes[i].v1 += origin;
            shapes[i].v2 *= radius;
            shapes[i].v2 += origin;
            shapes[i].v3 *= radius;
            shapes[i].v3 += origin;
        }
    }

    template<size_t n> void addObjToScene(std::vector<shape_t>& shapes,              // добавить объект в сцену
        std::vector<light_t>& lights,
        const std::array<triangle_t, n>& obj,
        const input::object_props& props) {

        std::set<vector_t> diodes;
        for (auto& triangle : obj) {
            for (auto& edge : triangle.exportEdges()) {
                vector_t dv = edge.b - edge.a;
                double dk = 1.0 / (props.diodes_count - 1);
                for (size_t i = 0; i < props.diodes_count; ++i) {
                    diodes.insert(edge.a + i * dk * dv);
                }
            }
            shape_t shape;
            shape.c = props.color;
            shape.t = triangle;
            shape.r = props.ref;
            shapes.push_back(std::move(shape));
        }

        for (auto& diode : diodes) {
            light_t light;
            light.c = { 0.002,0.002,0.002 };
            light.p = diode;
            light.t = light_kind::diode;
            lights.push_back(std::move(light));
        }
    }

    void addFloorToScene(std::vector<shape_t>& shapes,                              // добавить пол в сцену
        const input::floor& floor_params) {
        auto floor = object_presets::rectangle(
            floor_params.w, floor_params.h);
        transformObj(floor, floor_params.origin, 1);

        for (auto& triangle : floor) {
            shape_t shape;
            shape.c = floor_params.color;
            shape.r = floor_params.ref;
            shape.t = triangle;
            shapes.push_back(std::move(shape));
        }

    }

}

int main(int argc, char** argv) {

    bool use_gpu = false;
    if (argc < 2) {
        std::cout << info::getOptimalConfig() << std::endl;
        return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "--default") {
        return 0;
    }
    else if (argc == 2 && std::string(argv[1]) == "--gpu") {
        use_gpu = false;
    }
    else if (argc == 2 && std::string(argv[1]) == "--cpu") {
        use_gpu = false;
    }
    else {
        std::cerr << "--gpu     : use gpu as a computing device\n";
        std::cerr << "--cpu     : use cpu as a computing device\n";
        std::cerr << "--default : show optimal configuration\n";
        return -1;
    }

    std::vector<shape_t> shapes;
    std::vector<light_t> lights;



    size_t frames;
    input::filename_template filename_tpl;
    int image_width, image_height;                              // ширина и высота изображения
    double view_angle;                                          // угол обзора камеры
    int depth, ssaa_ratio;                                      // глубина луча и уровень сглаживания


    std::cin >> frames >> filename_tpl;
    std::cin >> image_width >> image_height >> view_angle;      // ширина, высота, угол обзора камеры

    input::point_movement cam; std::cin >> cam;                 // движение камеры
    input::point_movement foc; std::cin >> foc;                 // движение фокуса

    {
        input::object_props props; std::cin >> props;           // додекаэдр
        auto obj = object_presets::dodecahedron();
        scene::transformObj(obj, props.origin, props.radius);
        scene::addObjToScene(shapes, lights, obj, props);
    }

    {
        input::object_props props; std::cin >> props;           // гексаэдр
        auto obj = object_presets::hexaeder();
        scene::transformObj(obj, props.origin, props.radius);
        scene::addObjToScene(shapes, lights, obj, props);
    }

    {
        input::object_props props; std::cin >> props;           // октаэдр
        auto obj = object_presets::octaedr();
        scene::transformObj(obj, props.origin, props.radius);
        scene::addObjToScene(shapes, lights, obj, props);
    }

    {
        input::floor floor_params; std::cin >> floor_params;    // пол сцены
        scene::addFloorToScene(shapes, floor_params);
    }

    {
        light_t ambient_light;                                  // фоновое освещение
        ambient_light.t = light_kind::ambient;
        std::cin >> ambient_light.c.r;
        std::cin >> ambient_light.c.g;
        std::cin >> ambient_light.c.b;
        lights.push_back(ambient_light);

        light_t point_light;                                    // точечный источник
        point_light.t = light_kind::point;
        std::cin >> point_light.p;
        std::cin >> point_light.c.r;
        std::cin >> point_light.c.g;
        std::cin >> point_light.c.b;
        lights.push_back(point_light);
    }
    std::cin >> depth >> ssaa_ratio;


    // константы
    const double pi = std::acos(-1);                            // PI
    const double dt = 2.0 * pi / frames;                        // временной шаг
    view_angle = pi * view_angle / 180;                         // угол обзора

    const int32_t ssaa_width = image_width * ssaa_ratio;            // ширина перед сглаживанием
    const int32_t ssaa_height = image_height * ssaa_ratio;          // высота перед сглаживанием
    const int32_t ssaa_size = ssaa_width * ssaa_height;             // пикс перед сглаживанием
    const int32_t image_size = image_width * image_height;          // пикс после сглаживания

    const int32_t shapes_size = shapes.size();
    const int32_t lights_size = lights.size();

    std::vector<uchar4> image(image_size);


    if (use_gpu) {
        std::vector<uchar4> ssaa(ssaa_size);
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point stop;


        antialiasing::ssaa_context_t ssaa_ctx;
        ssaa_ctx.in = ssaa.data();
        ssaa_ctx.height = image_height;
        ssaa_ctx.height_ratio = ssaa_ratio;
        ssaa_ctx.width = image_width;
        ssaa_ctx.width_ratio = ssaa_ratio;
        ssaa_ctx.out = image.data();


        for (size_t f = 0; f < frames; ++f) {
            const double t = f * dt;
            start = std::chrono::high_resolution_clock::now();
            renders::cpuRender(ssaa.data(), ssaa_width, ssaa_height,
                shapes.data(), shapes_size,
                lights.data(), lights_size,
                depth, view_angle,
                cam.getLocation(t), foc.getLocation(t));
            antialiasing::ssaa_function((ssaa_ctx));
            stop = std::chrono::high_resolution_clock::now();

            std::cout << "CPU Frame [ " << f << " / "
                << frames << " ]" << " duration: "
                << std::chrono::duration_cast<
                std::chrono::milliseconds>
                (stop - start).count() << " ms\n";

            auto frame_name = filename_tpl.getFilename(f);

            std::fstream out(frame_name, std::ios::binary | std::ios::out);
            out.write(reinterpret_cast<char*>(&image_width),
                                            sizeof(int32_t));
            out.write(reinterpret_cast<char*>(&image_height),
                                            sizeof(int32_t));
            out.write(reinterpret_cast<char*>(image.data()),
                                image_size * sizeof(uchar4));
        }
    }
    else {
        cudaEvent_t start_event, stop_event;
        HANDLE_ERROR(cudaEventCreate(&start_event));
        HANDLE_ERROR(cudaEventCreate(&stop_event));

        uchar4* dev_image = nullptr;
        uchar4* dev_ssaa = nullptr;
        shape_t* dev_shapes = nullptr;
        light_t* dev_lights = nullptr;

        HANDLE_ERROR(cudaMalloc((void**)(&dev_image), image_size * sizeof(uchar4)));
        HANDLE_ERROR(cudaMalloc((void**)(&dev_ssaa), ssaa_size * sizeof(uchar4)));
        HANDLE_ERROR(cudaMalloc((void**)(&dev_shapes), shapes_size * sizeof(shape_t)));
        HANDLE_ERROR(cudaMalloc((void**)(&dev_lights), lights_size * sizeof(light_t)));

        HANDLE_ERROR(cudaMemcpy(dev_shapes, shapes.data(), 
            shapes_size * sizeof(shape_t), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_lights, lights.data(), 
            lights_size * sizeof(light_t), cudaMemcpyHostToDevice));

        antialiasing::ssaa_context_t ssaa_ctx;
        ssaa_ctx.in = dev_ssaa;
        ssaa_ctx.height = image_height;
        ssaa_ctx.height_ratio = ssaa_ratio;
        ssaa_ctx.width = image_width;
        ssaa_ctx.width_ratio = ssaa_ratio;
        ssaa_ctx.out = dev_image;


        for (size_t f = 0; f < frames; ++f) {
            float duration, total_duration = 0;
            const double t = f * dt;
            HANDLE_ERROR(cudaEventRecord(start_event));
            renders::gpuRender<<<CUDA_BLOCKS, CUDA_THREADS>>> (
                                    dev_ssaa, ssaa_width, ssaa_height,
                                    dev_shapes, shapes_size,
                                    dev_lights, lights_size,
                                    depth, view_angle,
                                    cam.getLocation(t),
                                    foc.getLocation(t));
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaEventRecord(stop_event));
            HANDLE_ERROR(cudaEventSynchronize(stop_event));
            HANDLE_ERROR(cudaEventElapsedTime(
                &duration, start_event, stop_event));
            total_duration += duration;

            HANDLE_ERROR(cudaEventRecord(start_event));
            antialiasing::ssaa_kernel<<<CUDA_BLOCKS, CUDA_THREADS>>> (
                                    ssaa_ctx);
            HANDLE_ERROR(cudaGetLastError());
            HANDLE_ERROR(cudaEventRecord(stop_event));
            HANDLE_ERROR(cudaEventSynchronize(stop_event));
            HANDLE_ERROR(cudaEventElapsedTime(
                &duration, start_event, stop_event));
            total_duration += duration;

            HANDLE_ERROR(cudaMemcpy(image.data(), dev_image,
                image_size * sizeof(uchar4), cudaMemcpyDeviceToHost));


            std::cout << "GPU Frame [ " << f << " / "
                << frames << " ]" << " duration: "
                << total_duration << " ms\n";

            auto frame_name = filename_tpl.getFilename(f);

            std::fstream out(frame_name, std::ios::binary | std::ios::out);

            out.write(reinterpret_cast<char*>(&image_width),
                                            sizeof(int32_t));
            out.write(reinterpret_cast<char*>(&image_height),
                                            sizeof(int32_t));

            out.write(reinterpret_cast<char*>(image.data()),
                               image_size * sizeof(uchar4));
        }

        HANDLE_ERROR(cudaFree(dev_image));
        HANDLE_ERROR(cudaFree(dev_lights));
        HANDLE_ERROR(cudaFree(dev_ssaa));
        HANDLE_ERROR(cudaFree(dev_shapes));

        HANDLE_ERROR(cudaEventDestroy(start_event));
        HANDLE_ERROR(cudaEventDestroy(stop_event));
    }
    return 0;
}