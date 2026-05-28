#include <iostream>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <omp.h>

using namespace std;

// Resolucion simulada
const int WIDTH  = 4000;
const int HEIGHT = 4000;

// RGB
struct Pixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

// ------------------------------------------------------------
// Generar imagen aleatoria
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
// Histograma usando atomic
// ------------------------------------------------------------
void histogramAtomic(const vector<Pixel>& image,
                     vector<int>& histogram)
{
    auto start =
        chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (long long i = 0; i < image.size(); i++)
    {
        int value = image[i].r;

        // Exclusión mutua
        #pragma omp atomic
        histogram[value]++;
    }

    auto end =
        chrono::high_resolution_clock::now();

    double time =
        chrono::duration<double>(end - start)
        .count();

    cout << "Tiempo Atomic: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// Histograma usando variables privadas
// ------------------------------------------------------------
void histogramPrivate(const vector<Pixel>& image,
                      vector<int>& histogram)
{
    auto start =
        chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        // Histograma privado por hilo
        vector<int> localHist(256, 0);

        #pragma omp for
        for (long long i = 0; i < image.size(); i++)
        {
            int value = image[i].r;

            localHist[value]++;
        }

        // Fusionar resultados
        #pragma omp critical
        {
            for (int i = 0; i < 256; i++)
            {
                histogram[i] += localHist[i];
            }
        }
    }

    auto end =
        chrono::high_resolution_clock::now();

    double time =
        chrono::duration<double>(end - start)
        .count();

    cout << "Tiempo Variables Privadas: "
         << time
         << " segundos\n";
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------
int main()
{
    vector<Pixel> image(WIDTH * HEIGHT);

    generateImage(image);

    vector<int> histogram1(256, 0);

    vector<int> histogram2(256, 0);

    cout << "===== HISTOGRAMA ATOMIC =====\n";

    histogramAtomic(image, histogram1);

    cout << "\n===== HISTOGRAMA PRIVADO =====\n";

    histogramPrivate(image, histogram2);

    return 0;
}
