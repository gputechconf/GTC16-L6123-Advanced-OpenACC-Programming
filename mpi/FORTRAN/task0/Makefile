all: run

laplace2d: laplace2d.F03 laplace2d_serial.F03 Makefile
	pgfortran -fast -acc -ta=tesla -Minfo=accel laplace2d.F03 laplace2d_serial.F03 -o laplace2d

clean:
	rm -f laplace2d laplace2d.solution *.o

run: laplace2d
	./laplace2d

profile: laplace2d
	nvprof -o laplace2d.nvvp ./laplace2d
