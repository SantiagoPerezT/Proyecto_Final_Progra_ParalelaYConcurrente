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

// Iteraciones Mandelbrot
const int MAX_ITER = 1000;

// RGB
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// ------------------------------------------------------------
// Guardar imagen PPM
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
// Generacion Mandelbrot con scheduler configurable
// ------------------------------------------------------------
void generateMandelbrot(vector<Pixel>& image,
                        omp_sched_t schedulerType,
                        int chunkSize,
                        const string& schedulerName)
{
    // Configurar scheduler
    omp_set_schedule(schedulerType, chunkSize);

    auto start =
        chrono::high_resolution_clock::now();

    // schedule(runtime) toma el scheduler configurado
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            // Conversion al plano complejo
            double real =
                (x - WIDTH / 2.0) * 4.0 / WIDTH;

            double imag =
                (y - HEIGHT / 2.0) * 4.0 / WIDTH;

            double zr = 0.0;
            double zi = 0.0;

            int iter = 0;

            // Formula Mandelbrot
            while ((zr * zr + zi * zi <= 4.0) &&
                    iter < MAX_ITER)
            {
                double temp =
                    zr * zr - zi * zi + real;

                zi = 2.0 * zr * zi + imag;

                zr = temp;

                iter++;
            }

            // Escala de grises
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

    cout << "Scheduler: "
         << schedulerName
         << " | Chunk: "
         << chunkSize
         << " | Tiempo: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    vector<Pixel> image(WIDTH * HEIGHT);

    cout << "===== STATIC =====\n";

    generateMandelbrot(
        image,
        omp_sched_static,
        1,
        "static"
    );

    generateMandelbrot(
        image,
        omp_sched_static,
        16,
        "static"
    );

    generateMandelbrot(
        image,
        omp_sched_static,
        64,
        "static"
    );

    cout << "\n===== DYNAMIC =====\n";

    generateMandelbrot(
        image,
        omp_sched_dynamic,
        1,
        "dynamic"
    );

    generateMandelbrot(
        image,
        omp_sched_dynamic,
        16,
        "dynamic"
    );

    generateMandelbrot(
        image,
        omp_sched_dynamic,
        64,
        "dynamic"
    );

    cout << "\n===== GUIDED =====\n";

    generateMandelbrot(
        image,
        omp_sched_guided,
        1,
        "guided"
    );

    generateMandelbrot(
        image,
        omp_sched_guided,
        16,
        "guided"
    );

    generateMandelbrot(
        image,
        omp_sched_guided,
        64,
        "guided"
    );

    // Guardar ultima imagen generada
    savePPM("mandelbrot.ppm", image);

    return 0;
}