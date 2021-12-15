#!/usr/bin/env bash

for r in 5 10; do
	for d in 500 1000; do
		echo "Trying d=${d}, r=${r}"
		julia --project=scripts scripts/bilinear_sensing.jl \
			--d ${d} --r ${r} --m $(( 3 * r * d )) --eps-decrease 0.5
	done
done

# Run special case for r = 1
for d in 1000 2500 5000; do
	julia --project=scripts scripts/bilinear_sensing.jl \
		--d ${d} --r 1 --m $(( 3 * d )) --eps-decrease 0.5
done
