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
  double st = omp_get_wtime();

  num_blocks = 8;
  if ( argc > 1 ) num_blocks = atoi(argv[1]);
  block_size = (HEIGHT/num_blocks)*WIDTH;

  acc_init(acc_device_nvidia);

#pragma acc data create(image[WIDTH*HEIGHT])
  for(int block = 0; block < num_blocks; block++ ) {
    int start = block * (HEIGHT/num_blocks),
        end   = start + (HEIGHT/num_blocks);
#pragma acc update device(image[block*block_size:block_size]) async(block%2)
#pragma acc parallel loop async(block%2)
    for(int y=start;y<end;y++) {
      for(int x=0;x<WIDTH;x++) {
        image[y*WIDTH+x]=mandelbrot(x,y);
      }
    }
#pragma acc update self(image[block*block_size:block_size]) async(block%2)
  }
#pragma acc wait
  
  double et = omp_get_wtime();
  printf("Time: %lf seconds.\n", (et-st));
  fwrite(image,sizeof(unsigned char),WIDTH*HEIGHT,fp);
  fclose(fp);
  free(image);
  return 0;
}
