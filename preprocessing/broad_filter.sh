#!/bin/bash

for i in {0..99}; do
  if [ "$i" -eq 20 ]; then continue; fi
  python preprocessing/broad_filter.py --batch_id=$i
done