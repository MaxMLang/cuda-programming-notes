//
// Pattern-matching program
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// include kernel function
////////////////////////////////////////////////////////////////////////

#include "match_kernel.cu"

////////////////////////////////////////////////////////////////////////
// declare Gold routine
////////////////////////////////////////////////////////////////////////

void gold_match(unsigned int *, unsigned int *, int *, int, int);

////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){

  char *ctext;
  // Using "const char *" to resolve compilation warnings about deprecated conversions.
  const char *cwords[] = {"cuti", "gold", "text", "word"};
  unsigned int  *text,  *words;

  int   length, len, nwords=4;
  int   cpu_matches[4]={0, 0, 0, 0};
  int   gpu_matches[4]={0, 0, 0, 0};


  // read in text for processing

  FILE *fp;
  fp = fopen("match.cu","r");

  length = 0;
  while (getc(fp) != EOF) length++;

  ctext = (char *) malloc(length+4);

  rewind(fp);

  for (int l=0; l<length; l++) ctext[l] = getc(fp);
  for (int l=length; l<length+4; l++) ctext[l] = ' ';

  fclose(fp);

  // define number of words of text, and set pointers

  len  = length/4;
  text = (unsigned int *) ctext;

  // define words for matching

  words = (unsigned int *) malloc(nwords*sizeof(unsigned int));

  for (int w=0; w<nwords; w++) {
    words[w] = ((unsigned int) cwords[w][0])
             + ((unsigned int) cwords[w][1])*256
             + ((unsigned int) cwords[w][2])*256*256
             + ((unsigned int) cwords[w][3])*256*256*256;
  }

  // CPU execution

  gold_match(text, words, cpu_matches, nwords, len);

  printf(" CPU matches = %d %d %d %d \n",
         cpu_matches[0],cpu_matches[1],cpu_matches[2],cpu_matches[3]);

  // GPU execution
  
  unsigned int *d_text, *d_words;
  int *d_matches;

  // Allocate memory on the GPU
  cudaMalloc((void **) &d_text, (len+1) * sizeof(unsigned int));
  cudaMalloc((void **) &d_words, nwords * sizeof(unsigned int));
  cudaMalloc((void **) &d_matches, nwords * sizeof(int));

  // Copy data from CPU to GPU
  cudaMemcpy(d_text, text, (len+1) * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_words, words, nwords * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_matches, gpu_matches, nwords * sizeof(int), cudaMemcpyHostToDevice); // Initialize with zeros

  // Launch the kernel with 1 block and nwords threads
  match_kernel<<<1, nwords>>>(d_text, d_words, d_matches, nwords, len);

  // Copy results from GPU back to CPU
  cudaMemcpy(gpu_matches, d_matches, nwords * sizeof(int), cudaMemcpyDeviceToHost);

  printf(" GPU matches = %d %d %d %d \n",
         gpu_matches[0],gpu_matches[1],gpu_matches[2],gpu_matches[3]);


  // Release GPU and CPU memory
  cudaFree(d_text);
  cudaFree(d_words);
  cudaFree(d_matches);

  free(ctext);
  free(words);
}
