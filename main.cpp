#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// Resolucion
const int WIDTH  = 3840;
const int HEIGHT = 2160;

// RGB
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// ------------------------------------------------------------
// Generar imagen de prueba
// ------------------------------------------------------------
void generateImage(vector<Pixel>& image)
{
    #pragma omp parallel for
    for (long long i = 0; i < image.size(); i++)
    {
        image[i].r = rand() % 256;
        image[i].g = rand() % 256;
        image[i].b = rand() % 256;
    }
}

// ------------------------------------------------------------
// Filtro Gaussiano usando SIMD
// ------------------------------------------------------------
void gaussianBlurSIMD(const vector<Pixel>& input,
                      vector<Pixel>& output)
{
    const int radius = 5;

    const int kernelSize =
        radius * 2 + 1;

    float kernel[11][11];

    float sigma = 5.0f;

    float sum = 0.0f;

    // Crear kernel
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            float value =
                exp(-(x*x + y*y)
                / (2 * sigma * sigma));

            kernel[y + radius][x + radius]
                = value;

            sum += value;
        }
    }

    // Normalizar kernel
    for (int y = 0; y < kernelSize; y++)
    {
        for (int x = 0; x < kernelSize; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    auto start =
        chrono::high_resolution_clock::now();

    // SPMD
    #pragma omp parallel for
    for (int y = radius;
         y < HEIGHT - radius;
         y++)
    {
        for (int x = radius;
             x < WIDTH - radius;
             x++)
        {
            float r = 0.0f;
            float g = 0.0f;
            float b = 0.0f;

            // SIMD
            #pragma omp simd reduction(+:r,g,b)
            for (int ky = -radius;
                 ky <= radius;
                 ky++)
            {
                for (int kx = -radius;
                     kx <= radius;
                     kx++)
                {
                    Pixel p =
                        input[(y + ky)
                        * WIDTH + (x + kx)];

                    float weight =
                        kernel[ky + radius]
                              [kx + radius];

                    r += p.r * weight;
                    g += p.g * weight;
                    b += p.b * weight;
                }
            }

            output[y * WIDTH + x] =
            {
                (unsigned char)r,
                (unsigned char)g,
                (unsigned char)b
            };
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    double time =
        chrono::duration<double>(end - start)
        .count();

    cout << "Tiempo SIMD: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    vector<Pixel> image(WIDTH * HEIGHT);

    vector<Pixel> blurred(WIDTH * HEIGHT);

    generateImage(image);

    gaussianBlurSIMD(image, blurred);

    return 0;
}
