


#include <stdio.h>
#include <map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include "test_util.hpp"







bool                    g_verbose = false;  

bool                    g_report = false;   











template <int NUM_BINS, int ACTIVE_CHANNELS>
inline void DecodePixel(float4 &pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;

        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}



template <int NUM_BINS, int ACTIVE_CHANNELS>
inline void DecodePixel(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char samples[4];
    samples[0] = pixel.x;
    samples[1] = pixel.y;
    samples[2] = pixel.z;
    samples[3] = pixel.w;


        for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}



template <int NUM_BINS, int ACTIVE_CHANNELS>
inline void DecodePixel(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel;
}

#include "histogram_gmem_atomics.hpp"
#include "histogram_smem_atomics.hpp"

struct less_than_value
{
    inline bool operator()(
        const std::pair<std::string, double> &a,
        const std::pair<std::string, double> &b)
    {
        return a.second < b.second;
    }
};











struct TgaHeader
{
    char idlength;
    char colormaptype;
    char datatypecode;
    short colormaporigin;
    short colormaplength;
    char colormapdepth;
    short x_origin;
    short y_origin;
    short width;
    short height;
    char bitsperpixel;
    char imagedescriptor;

    void Parse (FILE *fptr)
    {
        idlength = fgetc(fptr);
        colormaptype = fgetc(fptr);
        datatypecode = fgetc(fptr);
        fread(&colormaporigin, 2, 1, fptr);
        fread(&colormaplength, 2, 1, fptr);
        colormapdepth = fgetc(fptr);
        fread(&x_origin, 2, 1, fptr);
        fread(&y_origin, 2, 1, fptr);
        fread(&width, 2, 1, fptr);
        fread(&height, 2, 1, fptr);
        bitsperpixel = fgetc(fptr);
        imagedescriptor = fgetc(fptr);
    }

    void Display (FILE *fptr)
    {
        fprintf(fptr, "ID length:           %d\n", idlength);
        fprintf(fptr, "Color map type:      %d\n", colormaptype);
        fprintf(fptr, "Image type:          %d\n", datatypecode);
        fprintf(fptr, "Color map offset:    %d\n", colormaporigin);
        fprintf(fptr, "Color map length:    %d\n", colormaplength);
        fprintf(fptr, "Color map depth:     %d\n", colormapdepth);
        fprintf(fptr, "X origin:            %d\n", x_origin);
        fprintf(fptr, "Y origin:            %d\n", y_origin);
        fprintf(fptr, "Width:               %d\n", width);
        fprintf(fptr, "Height:              %d\n", height);
        fprintf(fptr, "Bits per pixel:      %d\n", bitsperpixel);
        fprintf(fptr, "Descriptor:          %d\n", imagedescriptor);
    }
};




void ParseTgaPixel(uchar4 &pixel, unsigned char *tga_pixel, int bytes)
{
    if (bytes == 4)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = tga_pixel[3];
    }
    else if (bytes == 3)
    {
        pixel.x = tga_pixel[2];
        pixel.y = tga_pixel[1];
        pixel.z = tga_pixel[0];
        pixel.w = 0;
    }
    else if (bytes == 2)
    {
        pixel.x = (tga_pixel[1] & 0x7c) << 1;
        pixel.y = ((tga_pixel[1] & 0x03) << 6) | ((tga_pixel[0] & 0xe0) >> 2);
        pixel.z = (tga_pixel[0] & 0x1f) << 3;
        pixel.w = (tga_pixel[1] & 0x80);
    }
}




void ReadTga(uchar4* &pixels, int &width, int &height, const char *filename)
{
    

    FILE *fptr;
    if ((fptr = fopen(filename, "rb")) == NULL)
    {
        fprintf(stderr, "File open failed\n");
        exit(-1);
    }

    

    TgaHeader header;
    header.Parse(fptr);


    width = header.width;
    height = header.height;

    

    if (header.datatypecode != 2 && header.datatypecode != 10)
    {
        fprintf(stderr, "Can only handle image type 2 and 10\n");
        exit(-1);
    }
    if (header.bitsperpixel != 16 && header.bitsperpixel != 24 && header.bitsperpixel != 32)
    {
        fprintf(stderr, "Can only handle pixel depths of 16, 24, and 32\n");
        exit(-1);
    }
    if (header.colormaptype != 0 && header.colormaptype != 1)
    {
        fprintf(stderr, "Can only handle color map types of 0 and 1\n");
        exit(-1);
    }

    

    int skip_bytes = header.idlength + (header.colormaptype * header.colormaplength);
    fseek(fptr, skip_bytes, SEEK_CUR);

    

    int pixel_bytes = header.bitsperpixel / 8;

    

    size_t image_bytes = width * height * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }
    memset(pixels, 0, image_bytes);

    

    unsigned char   tga_pixel[5];
    int             current_pixel = 0;
    while (current_pixel < header.width * header.height)
    {
        if (header.datatypecode == 2)
        {
            

            if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d  (uncompressed)\n", current_pixel);
                exit(-1);
            }
            ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
            current_pixel++;
        }
        else if (header.datatypecode == 10)
        {
            

            if (fread(tga_pixel, 1, pixel_bytes + 1, fptr) != pixel_bytes + 1)
            {
                fprintf(stderr, "Unexpected end of file at pixel %d (compressed)\n", current_pixel);
                exit(-1);
            }
            int run_length = tga_pixel[0] & 0x7f;
            ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
            current_pixel++;

            if (tga_pixel[0] & 0x80)
            {
                

                for (int i = 0; i < run_length; i++)
                {
                    ParseTgaPixel(pixels[current_pixel], &(tga_pixel[1]), pixel_bytes);
                    current_pixel++;
                }
            }
            else
            {
                

                for (int i = 0; i < run_length; i++)
                {
                    if (fread(tga_pixel, 1, pixel_bytes, fptr) != pixel_bytes)
                    {
                        fprintf(stderr, "Unexpected end of file at pixel %d (normal)\n", current_pixel);
                        exit(-1);
                    }
                    ParseTgaPixel(pixels[current_pixel], tga_pixel, pixel_bytes);
                    current_pixel++;
                }
            }
        }
    }

    

    fclose(fptr);
}












void GenerateRandomImage(uchar4* &pixels, int width, int height, int entropy_reduction)
{
    int num_pixels = width * height;
    size_t image_bytes = num_pixels * sizeof(uchar4);
    if ((pixels == NULL) && ((pixels = (uchar4*) malloc(image_bytes)) == NULL))
    {
        fprintf(stderr, "malloc of image failed\n");
        exit(-1);
    }

    for (int i = 0; i < num_pixels; ++i)
    {
        RandomBits(pixels[i].x, entropy_reduction);
        RandomBits(pixels[i].y, entropy_reduction);
        RandomBits(pixels[i].z, entropy_reduction);
        RandomBits(pixels[i].w, entropy_reduction);
    }
}












template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(float4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    float* samples = reinterpret_cast<float*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL] * float(NUM_BINS));
}



template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar4 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    unsigned char* samples = reinterpret_cast<unsigned char*>(&pixel);

    for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
        bins[CHANNEL] = (unsigned int) (samples[CHANNEL]);
}



template <int NUM_BINS, int ACTIVE_CHANNELS>
void DecodePixelGold(uchar1 pixel, unsigned int (&bins)[ACTIVE_CHANNELS])
{
    bins[0] = (unsigned int) pixel;
}




template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void HistogramGold(PixelType *image, int width, int height, unsigned int* hist)
{
    memset(hist, 0, ACTIVE_CHANNELS * NUM_BINS * sizeof(unsigned int));

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            PixelType pixel = image[i + j * width];

            unsigned int bins[ACTIVE_CHANNELS];
            DecodePixelGold<NUM_BINS>(pixel, bins);

            for (int CHANNEL = 0; CHANNEL < ACTIVE_CHANNELS; ++CHANNEL)
            {
                hist[(NUM_BINS * CHANNEL) + bins[CHANNEL]]++;
            }
        }
    }
}











template <
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void RunTest(
    std::vector<std::pair<std::string, double> >&   timings,
    PixelType*                                      pixels,
    const int                                       width,
    const int                                       height,
    unsigned int *                                  d_hist,
    unsigned int *                                  h_hist,
    int                                             timing_iterations,
    const char *                                    long_name,
    const char *                                    short_name,
    double (*f)(PixelType*, 
                int,   

                int,   

                unsigned int*, 
                bool)
    )
{
    if (!g_report) printf("%s ", long_name); fflush(stdout);

    

    (*f)(pixels, width, height, d_hist, !g_report);

    int compare = CompareDeviceResults(h_hist, d_hist, ACTIVE_CHANNELS * NUM_BINS, true, g_verbose);
    if (!g_report) printf("\t%s\n", compare ? "FAIL" : "PASS"); fflush(stdout);

    double elapsed_ms = 0;
    for (int i = 0; i < timing_iterations; i++)
    {
        elapsed_ms += (*f)(pixels, width, height, d_hist, false);
    }
    double avg_us = (elapsed_ms / timing_iterations) * 1000;    

    timings.push_back(std::pair<std::string, double>(short_name, avg_us));

    if (!g_report)
    {
        printf("Avg time %.3f us (%d iterations)\n", avg_us, timing_iterations); fflush(stdout);
    }
    else
    {
        printf("%.3f, ", avg_us); fflush(stdout);
    }

    

}




template <
    int         NUM_CHANNELS,
    int         ACTIVE_CHANNELS,
    int         NUM_BINS,
    typename    PixelType>
void TestMethods(
    PixelType*  h_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    size_t pixel_bytes = width * height * sizeof(PixelType);
    if (g_report) printf("%.3f, ", double(pixel_bytes) / bandwidth_GBs / 1000);

    

    size_t histogram_bytes = NUM_BINS * ACTIVE_CHANNELS * sizeof(unsigned int);
    unsigned int *h_hist = (unsigned int *) malloc(histogram_bytes);
    unsigned int *d_hist = (unsigned int *) malloc(histogram_bytes);


    

    HistogramGold<ACTIVE_CHANNELS, NUM_BINS>(h_pixels, width, height, h_hist);

    

    std::vector<std::pair<std::string, double> > timings;

    

    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, h_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Shared memory atomics", "smem atomics", run_smem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);
    RunTest<ACTIVE_CHANNELS, NUM_BINS>(timings, h_pixels, width, height, d_hist, h_hist, timing_iterations,
        "Global memory atomics", "gmem atomics", run_gmem_atomics<ACTIVE_CHANNELS, NUM_BINS, PixelType>);

    

    if (!g_report)
    {
        std::sort(timings.begin(), timings.end(), less_than_value());
        printf("Timings (us):\n");
        for (int i = 0; i < timings.size(); i++)
        {
            double bandwidth = height * width * sizeof(PixelType) / timings[i].second / 1000;
            printf("\t %.3f %s (%.3f GB/s, %.3f%% peak)\n", timings[i].second, timings[i].first.c_str(), bandwidth, bandwidth / bandwidth_GBs * 100);
        }
        printf("\n");
    }

    

    free(h_hist);
    free(d_hist);
}




void TestGenres(
    uchar4*     uchar4_pixels,
    int         height,
    int         width,
    int         timing_iterations,
    double      bandwidth_GBs)
{
    int num_pixels = width * height;

    {
        if (!g_report) printf("1 channel uchar1 tests (256-bin):\n\n"); fflush(stdout);

        size_t      image_bytes     = num_pixels * sizeof(uchar1);
        uchar1*     uchar1_pixels   = (uchar1*) malloc(image_bytes);

        

        for (int i = 0; i < num_pixels; ++i)
        {
            uchar1_pixels[i] = (unsigned char)
                (((unsigned int) uchar4_pixels[i].x +
                  (unsigned int) uchar4_pixels[i].y +
                  (unsigned int) uchar4_pixels[i].z) / 3);
        }

        TestMethods<1, 1, 256>(uchar1_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(uchar1_pixels);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel uchar4 tests (256-bin):\n\n"); fflush(stdout);
        TestMethods<4, 3, 256>(uchar4_pixels, width, height, timing_iterations, bandwidth_GBs);
        if (g_report) printf(", ");
    }

    {
        if (!g_report) printf("3/4 channel float4 tests (256-bin):\n\n"); fflush(stdout);
        size_t      image_bytes     = num_pixels * sizeof(float4);
        float4*     float4_pixels   = (float4*) malloc(image_bytes);

        

        for (int i = 0; i < num_pixels; ++i)
        {
            float4_pixels[i].x = float(uchar4_pixels[i].x) / 256;
            float4_pixels[i].y = float(uchar4_pixels[i].y) / 256;
            float4_pixels[i].z = float(uchar4_pixels[i].z) / 256;
            float4_pixels[i].w = float(uchar4_pixels[i].w) / 256;
        }
        TestMethods<4, 3, 256>(float4_pixels, width, height, timing_iterations, bandwidth_GBs);
        free(float4_pixels);
        if (g_report) printf("\n");
    }
}




int main(int argc, char **argv)
{
    

    CommandLineArgs args(argc, argv);
    if (args.CheckCmdLineFlag("help"))
    {
        printf(
            "%s "
            "[--v] "
            "[--i=<timing iterations>] "
            "\n\t"
                "--file=<.tga filename> "
            "\n\t"
                "--entropy=<-1 (0%%), 0 (100%%), 1 (81%%), 2 (54%%), 3 (34%%), 4 (20%%), ..."
                "[--height=<default: 1080>] "
                "[--width=<default: 1920>] "
            "\n", argv[0]);
        exit(0);
    }

    std::string         filename;
    int                 timing_iterations   = 100;
    int                 entropy_reduction   = 0;
    int                 height              = 1080;
    int                 width               = 1920;

    g_verbose = args.CheckCmdLineFlag("v");
    g_report = args.CheckCmdLineFlag("report");
    args.GetCmdLineArgument("i", timing_iterations);
    args.GetCmdLineArgument("file", filename);
    args.GetCmdLineArgument("height", height);
    args.GetCmdLineArgument("width", width);
    args.GetCmdLineArgument("entropy", entropy_reduction);

    

    args.DeviceInit();

    

    

    

    

    

    

    
    double bandwidth_GBs = 41;  


    

    uchar4* uchar4_pixels = NULL;
    if (!g_report)
    {
        if (!filename.empty())
        {
            

            ReadTga(uchar4_pixels, width, height, filename.c_str());
            printf("File %s: width(%d) height(%d)\n\n", filename.c_str(), width, height); fflush(stdout);
        }
        else
        {
            

            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            printf("Random image: entropy-reduction(%d) width(%d) height(%d)\n\n", entropy_reduction, width, height); fflush(stdout);
        }

        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
    }
    else
    {
        

        printf("Test, MIN, RLE CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM, , MIN, RLE_CUB, SMEM, GMEM\n");

        

        for (entropy_reduction = 0; entropy_reduction < 5; ++entropy_reduction)
        {
            printf("entropy reduction %d, ", entropy_reduction);
            GenerateRandomImage(uchar4_pixels, width, height, entropy_reduction);
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
        printf("entropy reduction -1, ");
        GenerateRandomImage(uchar4_pixels, width, height, -1);
        TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        printf("\n");

        

        std::vector<std::string> file_tests;
        file_tests.push_back("animals");
        file_tests.push_back("apples");
        file_tests.push_back("sunset");
        file_tests.push_back("cheetah");
        file_tests.push_back("nature");
        file_tests.push_back("operahouse");
        file_tests.push_back("austin");
        file_tests.push_back("cityscape");

        for (int i = 0; i < file_tests.size(); ++i)
        {
            printf("%s, ", file_tests[i].c_str());
            std::string filename = std::string("histogram/benchmark/") + file_tests[i] + ".tga";
            ReadTga(uchar4_pixels, width, height, filename.c_str());
            TestGenres(uchar4_pixels, height, width, timing_iterations, bandwidth_GBs);
        }
    }

    free(uchar4_pixels);

    printf("\n\n");

    return 0;
}