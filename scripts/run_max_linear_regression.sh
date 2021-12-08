#!/usr/bin/env bash

for k in 2 4 8
do
	for d in 50 100 200
	do
		echo "Trying (d,k)=(${d},${k})"
		julia --project=scripts scripts/max_linear_regression.jl \
			--k ${k} --m $(( 4 * d * k )) --d ${d} --show-amortized
	done
done
