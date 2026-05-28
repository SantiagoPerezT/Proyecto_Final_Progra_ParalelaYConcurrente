#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>

using namespace std;

// Resolucion ultra HD (8K)
const int WIDTH  = 7680;
const int HEIGHT = 4320;

// Numero maximo de iteraciones Mandelbrot
const int MAX_ITER = 1000;

// Estructura RGB
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// ------------------------------------------------------------
// Guardar imagen en formato PPM
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
// Generar fractal Mandelbrot (SECUENCIAL)
// ------------------------------------------------------------
void generateMandelbrot(vector<Pixel>& image)
{
    auto start =
        chrono::high_resolution_clock::now();

    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            // Conversion de pixeles al plano complejo
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

    cout << "Tiempo Mandelbrot: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// Aplicar filtro Gaussiano pesado
// ------------------------------------------------------------
void gaussianBlur(const vector<Pixel>& input,
                  vector<Pixel>& output)
{
    // Radio grande = convolucion pesada
    const int radius = 5;

    const int kernelSize =
        radius * 2 + 1;

    float kernel[11][11];

    float sigma = 5.0f;

    float sum = 0.0f;

    // Crear kernel gaussiano
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

    // Aplicar convolucion
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

            // Recorrer kernel
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

    cout << "Tiempo Convolucion: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    // Imagen original
    vector<Pixel> image(WIDTH * HEIGHT);

    // Imagen procesada
    vector<Pixel> blurred(WIDTH * HEIGHT);

    cout << "Generando Mandelbrot...\n";

    generateMandelbrot(image);

    cout << "Aplicando filtro Gaussiano...\n";

    gaussianBlur(image, blurred);

    cout << "Guardando imagen...\n";

    savePPM("mandelbrot.ppm", blurred);

    cout << "Proceso terminado\n";

    return 0;
}