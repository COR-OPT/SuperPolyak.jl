#!/usr/bin/env bash

for k in 5 10 20; do
	echo "Trying k=${k}"
	echo "1: Compressed sensing"
	julia -O3 --project=scripts scripts/compressed_sensing.jl \
		--k ${k} --m $(( 10 * k )) --d $(( 100 * k )) --eta-lb 0.75 \
		--eps-decrease 0.5
	echo "2: LASSO regression"
	julia -O3 --project=scripts scripts/lasso_regression.jl \
		--k ${k} --m $(( 10 * k )) --d $(( 100 * k )) --eta-lb 0.25 \
		--eps-decrease 0.5
done
