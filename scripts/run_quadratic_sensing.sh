#!/usr/bin/env bash

for r in 5 10; do
	for d in 500 1000; do
		julia -O3 --project=scripts scripts/quadratic_sensing.jl \
			--d ${d} --r ${r} --m $(( 3 * d * r )) --eps-decrease 0.5
	done
done
