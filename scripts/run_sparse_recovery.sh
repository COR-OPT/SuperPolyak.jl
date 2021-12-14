#!/usr/bin/env bash

for k in 5 10 20; do
	echo "Trying k=${k}"
	echo "1: Compressed sensing"
	julia --project=scripts scripts/compressed_sensing.jl \
		--k ${k} --m $(( 10 * k )) --d $(( 50 * k )) --eta-lb 0.25 \
		--show-amortized
	echo "2: LASSO regression"
	julia --project=scripts scripts/lasso_regression.jl \
		--k ${k} --m $(( 10 * k )) --d $(( 50 * k )) --eta-lb 0.25 \
		--show-amortized
done
