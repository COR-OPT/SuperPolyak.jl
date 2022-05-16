# SuperPolyak.jl

This repository contains a prototype implementation of the algorithms from the following paper:

V. Charisopoulos, D. Davis. *A superlinearly convergent subgradient method for sharp semismooth problems*, 2022. URL: https://arxiv.org/abs/2201.04611.

The implementation is available as a Julia package called `SuperPolyak.jl`, which can be embedded
in other Julia applications. The core algorithms can be found under `src/SuperPolyak.jl`.
Under `scripts/`, we have included all the scripts necessary to reproduce the numerical experiments in the
paper. We recommend running the code using Julia 1.6 or later.

## One-time setup

To install the dependencies of the package, run the following from the root directory
of this repository:

```shell
$ julia --project=. -e 'import Pkg; Pkg.instantiate()'
```

To use the `SuperPolyak` package in an interactive session, open the Julia prompt using

```shell
$ julia --project=.
```

and enter `import SuperPolyak` in the Julia prompt, as you would with any other Julia package.

To install locally, enter the following in a Julia session:

```julia
julia> ]    # Enter Pkg mode
(@v1.6) pkg> add https://github.com/COR-OPT/SuperPolyak.jl
```

## Running the experiments

The commands below assume that the root of this repository is the current directory.

First, install all the necessary dependencies using

```shell
julia --project=scripts -e 'import Pkg; Pkg.instantiate()'
```

All the experiments in the paper are contained in individual scripts.
For example, to solve a phase retrieval instance with dimension `d = 100` and `m = 250`
measurements, we run:

```shell
julia --project=scripts scripts/phase_retrieval.jl --d 100 --m 250
```

To view the help text, including a description of the available arguments, simply run:

```shell
julia --project=scripts scripts/phase_retrieval.jl --help
```

The subdirectory includes scripts for solving the following problems:

* Phase retrieval (using both the subgradient and the alternating projections method)
* Quadratic sensing
* Bilinear sensing
* Max-linear regression
* ReLU regression
* Compressed sensing (using the alternating projections method)
* LASSO regression (using the proximal gradient method)
