#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i, threads_number;

    /*Esta correcto*/
    omp_set_dynamic(true);
    #pragma omp parallel for simd
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    /*Funciona*/
    omp_set_dynamic(true);
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        threads_number = omp_get_num_threads();

        // Creamos un histograma privado para cada hilo
        int private_thread_hist[nbr_bin] = {0};

        //Calcular el histgrama de manera local
        #pragma omp for
        for ( i = 0; i < img_size; i ++){
            private_thread_hist[img_in[i]]++;
        }

        #pragma omp critical
        {
            for ( i = 0; i < nbr_bin; i ++){
                hist_out[i] += private_thread_hist[i];
            }
        }
    }
    
    /*for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }*/
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int *thread_cdf;
    int i, cdf, min, d, threads_number, thread_id, cdf_local, cdf_correction;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    
    /*while(min == 0){
        min = hist_in[i++];
    }*/

    #pragma omp parallel for
    for(i = 0; i < nbr_bin; i++){
        if (min == 0 && hist_in[i] != 0 ){
            #pragma omp critical
            min = hist_in[i];
        }
    }
    d = img_size - min;

    
    /*Sigo trabajando en este*/
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
    


    /* Get the result image */
    omp_set_dynamic(true);
    #pragma omp parallel for schedule(static)
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
    }
}



