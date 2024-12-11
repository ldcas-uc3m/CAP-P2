#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include "hist-equ.h"


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i, threads_number;

    #pragma omp parallel for schedule(static)
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    #pragma omp parallel for reduction(+: hist_out[:nbr_bin])
    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in,
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    
    #pragma omp parallel
    {
        int min_local = 0;

        #pragma omp parallel for simd
        for(i = 0; i < nbr_bin; i++){
            if (min_local == 0 && hist_in[i] != 0){
                min_local = hist_in[i];
            }
        }
        
        #pragma omp critical
        if (min == 0 || (min_local != 0 && min_local > min)){
            min = min_local;
        }
    }


    d = img_size - min;

    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
    }
    

    /* Get the result image */
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



