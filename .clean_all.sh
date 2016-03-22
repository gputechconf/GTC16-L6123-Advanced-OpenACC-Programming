#!/bin/bash
for f in $(find mpi pipelining -name Makefile ) ; do 
  cd $(dirname $f) 
  make clean 
  cd - 
done
