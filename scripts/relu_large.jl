using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

import Hadamard: ifwht
import ReverseDiff: GradientTape, gradient!, compile

using SuperPolyak

include("util.jl")

struct ProblemInstance
  x::Vector{Float64}
  y::Vector{Float64}
  S::Matrix{Int}
end

function opA(S::Matrix{Int}, v::AbstractVector{Float64})
  return vec(ifwht(S .* v, 1))
end

function opAT(S::Matrix{Int}, v::AbstractVector{Float64})
  d, k = size(S)
  return (S .* ifwht(reshape(v, d, k), 1)) * ones(k)
end

function loss(problem::ProblemInstance)
  y = problem.y
  m = length(y)
  return z -> (1 / m) * norm(max.(opA(problem.S, z), 0.0) .- y, 1)
end

function grad(problem::ProblemInstance)
  y = problem.y
  S = problem.S
  m = length(y)
  grad_fn(z) = begin
    Ax = opA(S, z)
    return (1 / m) * opAT(S, sign.(max.(Ax, 0.0) .- y) .* (Ax .≥ 0))
  end
  return grad_fn
end

function initializer(problem::ProblemInstance, δ::Float64)
  d = length(problem.x)
  return problem.x + δ * normalize(randn(d))
end

function problem_instance(d, k)
  S = rand([-1, 1], d, k)
  x = normalize(randn(d))
  y = max.(opA(S, x), 0.0)
  return ProblemInstance(x, y, S)
end

function run_experiment(
  d,
  k,
  δ,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  bundle_system_solver,
  no_amortized,
  run_subgradient,
)
  problem = problem_instance(d, k)
  loss_fn = loss(problem)
  grad_fn = grad(problem)
  z = initializer(problem, 0.9)
  if run_subgradient
    @info "Running subgradient method..."
    _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
      SuperPolyak.subgradient_method(loss_fn, grad_fn, z, ϵ_tol)
    df_polyak = DataFrame(
      t = 1:length(loss_history_polyak),
      fvals = loss_history_polyak,
      cumul_oracle_calls = 0:oracle_calls_polyak,
      cumul_elapsed_time = cumsum(elapsed_time_polyak),
    )
    CSV.write("relu_large_$(d)_$(k)_polyak.csv", df_polyak)
  else
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
      "relu_large_$(d)_$(k)_bundle.csv",
      result,
      no_amortized,
    )
  end
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on ReLU regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The dimension of the unknown signal."
  default = 300
  "--k"
  arg_type = Int
  help = "The number of random binary masks."
  default = 3
  "--run-subgradient"
  help = "Set to run the subgradient method."
  action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["d"],
  args["k"],
  args["initial-distance"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["bundle-system-solver"],
  args["no-amortized"],
  args["run-subgradient"],
)
