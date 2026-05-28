#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <omp.h>

using namespace std;

// Resolucion 8K
const int WIDTH  = 7680;
const int HEIGHT = 4320;

// Iteraciones maximas
const int MAX_ITER = 1000;

// RGB
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// ------------------------------------------------------------
// Guardar imagen
// ------------------------------------------------------------
void savePPM(const string& filename,
             const vector<Pixel>& image)
{
    ofstream file(filename, ios::binary);

    file << "P6\n"
         << WIDTH << " "
         << HEIGHT
         << "\n255\n";

    for (const auto& pixel : image)
    {
        file.write((char*)&pixel, 3);
    }

    file.close();
}

// ------------------------------------------------------------
// Mandelbrot paralelo con OpenMP
// ------------------------------------------------------------
void generateMandelbrot(vector<Pixel>& image)
{
    auto start =
        chrono::high_resolution_clock::now();

    // Paralelizacion del bucle externo
    #pragma omp parallel for
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            double real =
                (x - WIDTH / 2.0) * 4.0 / WIDTH;

            double imag =
                (y - HEIGHT / 2.0) * 4.0 / WIDTH;

            double zr = 0.0;
            double zi = 0.0;

            int iter = 0;

            while ((zr * zr + zi * zi <= 4.0) &&
                    iter < MAX_ITER)
            {
                double temp =
                    zr * zr - zi * zi + real;

                zi = 2.0 * zr * zi + imag;

                zr = temp;

                iter++;
            }

            unsigned char color =
                (unsigned char)
                (255.0 * iter / MAX_ITER);

            image[y * WIDTH + x] =
            {
                color,
                color,
                color
            };
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    double time =
        chrono::duration<double>(end - start)
        .count();

    cout << "Tiempo Mandelbrot OpenMP: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// Convolucion Gaussiana paralela
// ------------------------------------------------------------
void gaussianBlur(const vector<Pixel>& input,
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

    // Normalizar
    for (int y = 0; y < kernelSize; y++)
    {
        for (int x = 0; x < kernelSize; x++)
        {
            kernel[y][x] /= sum;
        }
    }

    auto start =
        chrono::high_resolution_clock::now();

    // Paralelizacion del blur
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

    cout << "Tiempo Convolucion OpenMP: "
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

    cout << "Generando Mandelbrot...\n";

    generateMandelbrot(image);

    cout << "Aplicando filtro Gaussiano...\n";

    gaussianBlur(image, blurred);

    cout << "Guardando imagen...\n";

    savePPM("mandelbrot_openmp.ppm", blurred);

    cout << "Proceso terminado\n";

    return 0;
}