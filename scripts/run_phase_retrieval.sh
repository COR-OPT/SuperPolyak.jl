#!/usr/bin/env bash

for d in 1000 2500 5000; do
	echo "Trying d=${d}"
	julia -O3 --project=scripts scripts/phase_retrieval.jl \
		--d ${d} --m $(( 3 * d )) --eps-decrease 0.5
done
