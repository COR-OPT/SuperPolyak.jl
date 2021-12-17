#!/usr/bin/env bash

RUNCMD='julia -O3 --project=scripts scripts/bilinear_sensing.jl --m 15000 --eta-lb 0.5 --eps-decrease 0.5'

echo "Running d=500, r=10"
$(RUNCMD) --d 500 --r 10
echo "Running d=1000, r=5"
$(RUNCMD) --d 1000 --r 5
echo "Running d=2500, r=2"
$(RUNCMD) --d 2500 --r 2

# Run special case for r = 1
for d in 1000 2500 5000; do
	julia --project=scripts scripts/bilinear_sensing.jl \
		--d ${d} --r 1 --m $(( 3 * d )) --eps-decrease 0.5
done
