program mandelbrot_main
use mandelbrot_mod
use openacc
implicit none
integer      :: num_blocks
integer(1)   :: image(HEIGHT, WIDTH)
integer      :: iy, ix
integer      :: block, block_size, block_start
integer      :: starty, endy, gpu, num_gpus
real         :: startt, stopt
character(8) :: arg

num_gpus = acc_get_num_devices(acc_device_nvidia)
! This loop is for eating initialization overhead
do gpu=0,num_gpus-1
  call acc_init(acc_device_nvidia)
  call acc_set_device_num(gpu,acc_device_nvidia)
enddo

call getarg(1, arg)
num_blocks = 64
if ( arg /= '' ) then
  read(arg, '(I10)') num_blocks
endif
print *,'num_blocks',num_blocks
block_size = (HEIGHT*WIDTH)/num_blocks

image = 0

call cpu_time(startt)
#ifdef _OPENACC
do gpu=0,num_gpus-1
  call acc_set_device_num(gpu,acc_device_nvidia)
  !$acc enter data create(image)
enddo
#endif

do block=0,(num_blocks-1)
  starty = block  * (WIDTH/num_blocks) + 1
  endy   = min(starty + (WIDTH/num_blocks), WIDTH)
  ! Cycle through GPUs
  call acc_set_device_num(mod(block,num_gpus),acc_device_nvidia)
  !$acc parallel loop async
  do iy=starty,endy
    do ix=1,HEIGHT
      image(ix,iy) = min(max(int(mandelbrot(ix-1,iy-1)),0),MAXCOLORS)
    enddo
  enddo
  !$acc update self(image(:,starty:endy)) async
enddo
#ifdef _OPENACC
do gpu=0,num_gpus-1
  call acc_set_device_num(gpu,acc_device_nvidia)
  !$acc wait
  !$acc exit data delete(image)
enddo
#endif
call cpu_time(stopt)

print *,"Time:",(stopt-startt)

call write_pgm(image,'image.pgm')
end
