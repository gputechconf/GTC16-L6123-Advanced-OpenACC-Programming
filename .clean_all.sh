#!/bin/bash
for f in $(find . -name Makefile ) ; do 
  cd $(dirname $f) 
  make clean 
  cd - 
done
