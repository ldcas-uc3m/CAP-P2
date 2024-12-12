#include <iostream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "hist-equ.h"

void run_cpu_color_test(PPM_IMG img_in, int full_h);
void run_cpu_gray_test(PGM_IMG img_in, int full_h);



int main(){

    /* omp */
    omp_set_dynamic(0); // Deshabilitar ajuste dinamico de hilos
    omp_set_num_threads(omp_get_max_threads()); // Usar hilos configurados por OMP_NUM_THREADS

    /* mpi */

    int w_size;  // number of total processes
    int w_rank;  // process ID

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);


    /* grayscale */

    PGM_IMG img_ibuf_g;

    int width;
    int height;

    if (w_rank == 0) {
        printf("Running contrast enhancement for gray-scale images.\n");
        img_ibuf_g = read_pgm("in.pgm");
        width = img_ibuf_g.w;
        height = img_ibuf_g.h;

    }
    // broadcast og size
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // divide image
    std::vector<int> counts {};  // number of elements to send to each processor
    std::vector<int> displacements {};  // displacements for each processor

    int disp = 0;  // current displacement

    for (int i = 0; i < w_size; ++i) {
        displacements.push_back(disp);

        // compute counts
        int rows = height / w_size;

        if (i != 0) ++rows;  // add top row
        if (i != w_size - 1) ++rows;  // add bottom row

        // distribute remainder rows (if img height is not exactly divisible)
        if (i < height % w_size) ++rows;

        counts.push_back(rows * width);

        // update displacement
        disp += (rows - 2) * width;  // -2 bc of overlapping rows
    }

    std::cout << "Process " << w_rank << " computes: [" << displacements[w_rank] << ", " << displacements[w_rank] + counts[w_rank] - 1 << "] (" << counts[w_rank] << " elements, " << counts[w_rank] / width << "/" << height << " rows)\n";

    {  // trust me bro, this is HIGHLY EFFICIENT C++ code
        std::vector<unsigned char> rcv_buf_g {};
        rcv_buf_g.reserve(counts[w_rank]);

        // send img
        MPI_Scatterv(
            img_ibuf_g.img,
            counts.data(),
            displacements.data(),
            MPI_UNSIGNED_CHAR,
            rcv_buf_g.data(),
            counts[w_rank],
            MPI_UNSIGNED_CHAR,
            0,
            MPI_COMM_WORLD
        );

        // compose new local image
        img_ibuf_g.w = width;
        img_ibuf_g.h = counts[w_rank] / width;
        img_ibuf_g.img = rcv_buf_g.data();

        run_cpu_gray_test(img_ibuf_g, height);
    }  // ~rcv_buf_g()



    /* color */

    PPM_IMG img_ibuf_c;

    if (w_rank == 0) {
        printf("Running contrast enhancement for color images.\n");
        img_ibuf_c = read_ppm("in.ppm");
    }

    // assume img is same size as grayscale

    std::vector<unsigned char> rcv_buf_c_r {};
    rcv_buf_c_r.reserve(counts[w_rank]);

    std::vector<unsigned char> rcv_buf_c_g {};
    rcv_buf_c_g.reserve(counts[w_rank]);

    std::vector<unsigned char> rcv_buf_c_b {};
    rcv_buf_c_b.reserve(counts[w_rank]);

    // send R channel
    MPI_Scatterv(
        img_ibuf_c.img_r,
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        rcv_buf_c_r.data(),
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    img_ibuf_c.img_r = rcv_buf_c_r.data();

    // send G channel
    MPI_Scatterv(
        img_ibuf_c.img_g,
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        rcv_buf_c_g.data(),
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    img_ibuf_c.img_g = rcv_buf_c_g.data();

    // send B channel
    MPI_Scatterv(
        img_ibuf_c.img_b,
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        rcv_buf_c_b.data(),
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    img_ibuf_c.img_b = rcv_buf_c_b.data();

    img_ibuf_c.w = width;
    img_ibuf_c.h = counts[w_rank] / width;

    run_cpu_color_test(img_ibuf_c, height);

    MPI_Finalize();
    return 0;
}


void run_cpu_color_test(PPM_IMG img_in, int full_h)
{
    int w_size;  // number of total processes
    int w_rank;  // process ID

    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);


    double tstart;
    if (w_rank == 0) {
        printf("Starting CPU processing...\n");
        tstart = MPI_Wtime();
    }


    /* HSL */

    PPM_IMG img_obuf_hsl = contrast_enhancement_c_hsl(img_in, full_h);

    // join the image back

    std::vector<unsigned char> rcv_img_obuf_r {};
    rcv_img_obuf_r.reserve(img_obuf_hsl.w * full_h);

    std::vector<unsigned char> rcv_img_obuf_g {};
    rcv_img_obuf_g.reserve(img_obuf_hsl.w * full_h);

    std::vector<unsigned char> rcv_img_obuf_b {};
    rcv_img_obuf_b.reserve(img_obuf_hsl.w * full_h);

    std::vector<int> counts {};  // number of elements to send to each process
    std::vector<int> displacements {};  // displacements for each process

    int disp = 0;
    for (int i = 0; i < w_size; ++i) {
        displacements.push_back(disp);

        // counts
        int rows = full_h / w_size;
        if (i < full_h % w_size) ++rows;

        counts.push_back(rows * img_obuf_hsl.w);

        // update displacement
        disp += (rows * img_obuf_hsl.w);
    }

    // gather R channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_hsl.img_r : img_obuf_hsl.img_r + img_obuf_hsl.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_r.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_hsl.img_r = rcv_img_obuf_r.data();

    // gather G channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_hsl.img_g : img_obuf_hsl.img_g + img_obuf_hsl.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_g.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_hsl.img_g = rcv_img_obuf_g.data();

    // gather B channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_hsl.img_b : img_obuf_hsl.img_b + img_obuf_hsl.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_b.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_hsl.img_b = rcv_img_obuf_b.data();


    if (w_rank == 0) {
        double tfinish = MPI_Wtime();
        double totalTime = tfinish - tstart;

        img_obuf_hsl.h = full_h;

        printf("HSL Processing time: %f s\n", totalTime);

        write_ppm(img_obuf_hsl, "out_hsl.ppm");
    }


    /* YUV */

    PPM_IMG img_obuf_yuv = contrast_enhancement_c_yuv(img_in, full_h);

    // gather R channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_yuv.img_r : img_obuf_yuv.img_r + img_obuf_yuv.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_r.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_yuv.img_r = rcv_img_obuf_r.data();

    // gather G channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_yuv.img_g : img_obuf_yuv.img_g + img_obuf_yuv.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_g.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_yuv.img_g = rcv_img_obuf_g.data();

    // gather B channel
    MPI_Gatherv(
        w_rank == 0 ? img_obuf_yuv.img_b : img_obuf_yuv.img_b + img_obuf_yuv.w,
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf_b.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) img_obuf_yuv.img_b = rcv_img_obuf_b.data();


    if (w_rank == 0) {
        double tfinish = MPI_Wtime();
        double totalTime = tfinish - tstart;

        img_obuf_yuv.h = full_h;

        printf("YUV Processing time: %f s\n", totalTime);

        write_ppm(img_obuf_yuv, "out_yuv.ppm");
    }

}




void run_cpu_gray_test(PGM_IMG img_in, int full_h)
{
    int w_size;  // number of total processes
    int w_rank;  // process ID

    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);

    PGM_IMG img_obuf;

    double tstart;
    if (w_rank == 0) {
        printf("Starting CPU processing...\n");
        tstart = MPI_Wtime();
    }

    img_obuf = contrast_enhancement_g(img_in, full_h);

    // join the image back

    std::vector<unsigned char> rcv_img_obuf {};
    rcv_img_obuf.reserve(img_obuf.w * full_h);

    std::vector<int> counts {};  // number of elements to send to each process
    std::vector<int> displacements {};  // displacements for each process

    int disp = 0;
    for (int i = 0; i < w_size; ++i) {
        displacements.push_back(disp);

        // counts
        int rows = full_h / w_size;
        if (i < full_h % w_size) ++rows;

        counts.push_back(rows * img_obuf.w);

        // update displacement
        disp += (rows * img_obuf.w);
    }

    MPI_Gatherv(
        w_rank == 0 ? img_obuf.img : img_obuf.img + img_obuf.w,  // starting place: if != 0, we have to start one row below
        counts[w_rank],
        MPI_UNSIGNED_CHAR,
        rcv_img_obuf.data(),
        counts.data(),
        displacements.data(),
        MPI_UNSIGNED_CHAR,
        0,
        MPI_COMM_WORLD
    );

    if (w_rank == 0) {
        double tfinish = MPI_Wtime();
        double totalTime = tfinish - tstart;

        // re-save the full image
        img_obuf.h = full_h;
        img_obuf.img = rcv_img_obuf.data();

        printf("Processing time: %f s\n", totalTime);

        write_pgm(img_obuf, "out.pgm");
    }

    // free_pgm(img_obuf);
}  // ~rcv_img_obuf() (free(img_obuf.img))



PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];

    char *ibuf;
    PPM_IMG result;
    int v_max, i, mult;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));


    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    #pragma omp parallel for simd
    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }

    fclose(in_file);
    free(ibuf);

    return result;
}

void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i, mult;
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    #pragma omp parallel for simd
    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }

    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];


    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }

    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);


    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));


    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);
    fclose(in_file);

    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    free(img.img);
}

