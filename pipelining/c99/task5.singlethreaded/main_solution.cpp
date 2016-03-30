#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <omp.h>
#include <openacc.h>
#include "mandelbrot.h"
#include "constants.h"

using namespace std;

int main( int argc, char **argv ) {
  
  size_t bytes=WIDTH*HEIGHT*sizeof(unsigned int);
  unsigned char *image=(unsigned char*)malloc(bytes);
  int num_blocks, block_size;
  FILE *fp=fopen("image.pgm","wb");
  fprintf(fp,"P5\n%s\n%d %d\n%d\n","#comment",WIDTH,HEIGHT,MAX_COLOR);
  int gpu, num_gpus;

  num_gpus = acc_get_num_devices(acc_device_nvidia);
  printf("Found %d NVIDIA GPUs.\n", num_gpus);
  // This loop eats the overhead of device start-up to not affect timing
  for (gpu = 0; gpu < num_gpus; gpu++)
  {
    acc_init(acc_device_nvidia);
    acc_set_device_num(gpu,acc_device_nvidia);
  }

  num_blocks = 64;
  if ( argc > 1 ) num_blocks = atoi(argv[1]);
  block_size = (HEIGHT/num_blocks)*WIDTH;

  double st = omp_get_wtime();
#ifdef _OPENACC
  for (gpu = 0; gpu < num_gpus; gpu++)
  {
    acc_set_device_num(gpu,acc_device_nvidia);
#pragma acc enter data create(image[:bytes])
  }
#endif

  for(int block = 0; block < num_blocks; block++ ) {
    // Cycle through all GPUs
    acc_set_device_num(block%num_gpus,acc_device_nvidia);
    int start = block * (HEIGHT/num_blocks),
        end   = start + (HEIGHT/num_blocks);
#pragma acc parallel loop async
    for(int y=start;y<end;y++) {
      for(int x=0;x<WIDTH;x++) {
        image[y*WIDTH+x]=mandelbrot(x,y);
      }
    }
#pragma acc update self(image[block*block_size:block_size])
  }
#ifdef _OPENACC
  for (gpu = 0; gpu < num_gpus; gpu++)
  {
    acc_set_device_num(gpu,acc_device_nvidia);
#pragma acc wait
#pragma acc exit data delete(image)
  }
#endif
  
  double et = omp_get_wtime();
  printf("Time: %lf seconds.\n", (et-st));
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
