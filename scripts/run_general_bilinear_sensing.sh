#!/usr/bin/env bash

for r in 2 5; do
	for d in 75 150 300; do
		echo "Trying d=${d}, r=${r}"
		julia --project=scripts scripts/general_bilinear_sensing.jl \
			--d ${d} --r ${r} --m $(( 3 * d * r )) --show-amortized
	done
done
