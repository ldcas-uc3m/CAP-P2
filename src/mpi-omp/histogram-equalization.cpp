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
                            int * hist_in, int img_size, int nbr_bin, int min, int d){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    /* Construct the LUT by calculating the CDF */
    int cdf = 0;
    int i = 0;

    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*(nbr_bin - 1)/d + 0.5);
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



