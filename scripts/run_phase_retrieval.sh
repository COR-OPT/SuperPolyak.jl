#!/usr/bin/env bash

for d in 100 250 500; do
	echo "Trying d=${d}"
	julia --project=scripts scripts/phase_retrieval.jl \
		--d ${d} --m $(( 3 * d )) --show-amortized
done
