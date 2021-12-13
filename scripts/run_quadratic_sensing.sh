#!/usr/bin/env bash

for r in 1 5 10; do
	for d in 75 150 300; do
		echo "Trying d=${d}, r=${r}"
		julia --project=scripts scripts/quadratic_sensing.jl \
			--d ${d} --r ${r} --m $(( 2 * d * r )) --show-amortized
	done
done
