#!/usr/bin/env bash

for d in 100 250 500; do
	echo "Trying d=${d}"
	julia --project=scripts scripts/bilinear_sensing.jl \
		--d ${d} --m $(( 4 * d )) --show-amortized
done
