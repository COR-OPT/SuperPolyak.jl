#!/usr/bin/env bash

for k in 5 10 20; do
	echo "Trying k=${k}"
	julia --project=scripts scripts/compressed_sensing.jl \
		--k ${k} --m $(( 10 * k )) --d $(( 50 * k))
done
