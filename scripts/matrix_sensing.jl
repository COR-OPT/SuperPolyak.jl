using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

import Hadamard: ifwht

using SuperPolyak

include("util.jl")

struct ProblemInstance
  S₁::Matrix{Int}         # Binary masks for L.
  S₂::Matrix{Int}         # Binary masks for R.
  W::Matrix{Float64}
  X::Matrix{Float64}
  y::Vector{Float64}
end

"""
  hadm_mul(S::Matrix{Int}, v::AbstractVector{Float64})

Compute `Hv`, where `H = [H S₁; ... ; H Sₖ]` is a matrix drawn from the random
Hadamard ensemble.
"""
function hadm_mul(S::Matrix{Int}, v::AbstractVector{Float64})
  return vec(ifwht(S .* v, 1))
end

"""
  opA(S::Matrix{Int}, V::AbstractMatrix{Float64})

Compute the product:

  [H S₁; ...; H Sₖ] * V,

where `Sᵢ` is the diagonal matrix formed by the `i`-th column of `S`,
`H` is the `d × d` (unnormalized) Hadamard matrix, and `V` is an arbitrary
`d × r` matrix.
"""
function opA(S::Matrix{Int}, V::AbstractMatrix{Float64})
  d, k = size(S)
  # Flatten size `d × r × k` to `dk * r`
  return reshape(
    permutedims(ifwht(permutedims(S[:, :, :], [1, 3, 2]) .* V, 1), [1, 3, 2]),
    d * k,
    :,
  )
end

function hadm_mul_transpose(S::Matrix{Int}, v::AbstractVector{Float64})
  d, k = size(S)
  return (S .* ifwht(reshape(v, d, k), 1)) * ones(k)
end

"""
  opAT(S::Matrix{Int}, V::AbstractMatrix{Float64})

Compute the product:

  [S₁ H, ..., Sₖ H] * V,

where `Sᵢ` is the diagonal matrix formed by the `i`-th column of `S`,
`H` is the `d × d` (unnormalized) Hadamard matrix, and `V` is an arbitrary
`(dk) × r` matrix.
"""
function opAT(S::Matrix{Int}, V::AbstractMatrix{Float64})
  d, k = size(S)
  m, r = size(V)
  # Product [S_1 H, ..., S_k H] * [V_1; ... V_k]
  ATV =
    reshape(S, (d, 1, k)) .*
    (ifwht(permutedims(reshape(V', r, d, k), [2, 1, 3]), 1))
  return dropdims(sum(ATV, dims = 3), dims = 3)
end

function loss(problem::ProblemInstance)
  y = problem.y
  m = length(y)
  S₁ = problem.S₁
  S₂ = problem.S₂
  d, r = size(problem.W)
  # Preallocate W and X matrices.
  W = zeros(d, r)
  X = zeros(d, r)
  loss_fn(z) = begin
    # Separate the components.
    copyto!(W, 1, z, 1, d * r)
    copyto!(X, 1, z, (d * r + 1), d * r)
    # Compute row-wise product.
    return (1 / m) * norm(y .- sum(opA(S₁, W) .* opA(S₂, X), dims = 2)[:], 1)
  end
  return loss_fn
end

"""
  subgradient(problem::ProblemInstance)

Compute a subgradient of the robust ℓ₁ loss for a bilinear sensing `problem`
with measurement matrices drawn from the Hadamard ensemble.
"""
function subgradient(problem::ProblemInstance)
  y = problem.y
  m = length(y)
  S₁ = problem.S₁
  S₂ = problem.S₂
  d, r = size(problem.W)
  # Preallocate subgradient vector.
  gvec = zeros(2 * d * r)
  # Preallocate W and X matrices.
  W = zeros(d, r)
  X = zeros(d, r)
  grad_fn(z) = begin
    # Separate the components.
    copyto!(
      W,        # dest
      1,        # dest_offset
      z,        # src
      1,        # src_offset
      d * r,    # N
    )
    copyto!(
      X,            # dest
      1,            # dest_offset
      z,            # src
      (d * r + 1),  # src_offset
      d * r,        # N
    )
    # LW = [H S_1 W; ... H S_k W]. Similarly for RX.
    LW = opA(S₁, W)
    RX = opA(S₂, X)
    sg = sign.(sum(LW .* RX, dims = 2)[:] .- y)
    gvec[1:(d*r)] = (1 / m) * vec(opAT(S₁, sg .* RX))
    gvec[(d*r+1):end] = (1 / m) * vec(opAT(S₂, sg .* LW))
    return gvec
  end
  return grad_fn
end

function initializer(problem::ProblemInstance, δ::Float64)
  d, r = size(problem.W)
  return vec([problem.W problem.X]) + δ * normalize(randn(2 * d * r))
end

function problem_instance(d, r, k, κ)
  S₁ = rand([-1, 1], d, k)
  S₂ = rand([-1, 1], d, k)
  W = generate_conditioned_matrix(d, r, sqrt(κ))
  X = generate_conditioned_matrix(d, r, sqrt(κ))
  y = sum(opA(S₁, W) .* opA(S₂, X), dims = 2)[:]
  return ProblemInstance(S₁, S₂, W, X, y)
end

function run_experiment(
  d,
  r,
  k,
  κ,
  δ,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  bundle_system_solver,
  no_amortized,
  plot_inline,
)
  problem = problem_instance(d, r, k, κ)
  loss_fn = loss(problem)
  grad_fn = subgradient(problem)
  z = initializer(problem, δ)
  @info "Running subgradient method..."
  _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
    SuperPolyak.subgradient_method(loss_fn, grad_fn, z[:], ϵ_tol)
  df_polyak = DataFrame(
    t = 1:length(loss_history_polyak),
    fvals = loss_history_polyak,
    cumul_oracle_calls = 0:oracle_calls_polyak,
    cumul_elapsed_time = cumsum(elapsed_time_polyak),
  )
  CSV.write("matrix_sensing_$(d)_$(r)_$(k)_$(κ)_polyak.csv", df_polyak)
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    z,
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "matrix_sensing_$(d)_$(r)_$(k)_$(κ)_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo-")
    semilogy(0:oracle_calls_polyak, loss_history_polyak, "r-")
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
    legend(["SuperPolyak", "PolyakSGM"])
    show()
  end
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on ReLU regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The dimension of each column of the unknown signals."
  default = 300
  "--r"
  arg_type = Int
  help = "The number of columns of the unknown signals."
  default = 5
  "--k"
  arg_type = Int
  help = "The number of random binary masks."
  default = 3
  "--cond-num"
  arg_type = Int
  help = "The condition number of the problem."
  default = 1
  "--plot-inline"
  help = "Set to plot the results after running the algorithms."
  action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["d"],
  args["r"],
  args["k"],
  args["cond-num"],
  args["initial-distance"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["bundle-system-solver"],
  args["no-amortized"],
  args["plot-inline"],
)
