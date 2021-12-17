#!/usr/bin/env bash

RUNCMD='julia -O3 --project=scripts scripts/quadratic_sensing.jl --m 15000 --eta-lb 0.5 --eps-decrease 0.5'

echo "Running d=500, r=10"
$(RUNCMD) --d 500 --r 10
echo "Running d=1000, r=5"
$(RUNCMD) --d 1000 --r 5
echo "Running d=2500, r=2"
$(RUNCMD) --d 2500 --r 2
