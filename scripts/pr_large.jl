using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

import Hadamard: fwht
import ReverseDiff: GradientTape, gradient!, compile

using SuperPolyak

include("util.jl")

struct ProblemInstance
  x::Vector{Float64}
  y::Vector{Float64}
  s::Vector{Float64}
end

opA(problem::ProblemInstance, v::Vector{Float64}) = begin
  m = length(problem.y)
  s = problem.s
  return m * (fwht([s .* v; zeros(m - length(s))]))
end

opAT(problem::ProblemInstance, v::Vector{Float64}) = begin
  m = length(problem.y)
  d = length(problem.s)
  s = problem.s
  return s .* (m .* (fwht(v)[1:d]))
end

function loss(problem::ProblemInstance)
  y = problem.y
  m = length(y)
  return z -> (1 / m) * norm(abs.(opA(problem, z)) - problem.y, 1)
end

function grad(problem::ProblemInstance)
  y = problem.y
  s = problem.s
  m = length(y)
  d = length(s)
  grad_fn(z) = begin
    Ax = opA(problem, z)
    return (1 / m) * opAT(problem, sign.(abs.(Ax) .- y) .* sign.(Ax))
  end
  return grad_fn
end

function initializer(problem::ProblemInstance, δ::Float64)
  d = length(problem.x)
  return problem.x + δ * normalize(randn(d))
end

function problem_instance(m, d)
  s = rand([-1, 1], d)
  x = normalize(randn(d))
  y = abs.(m .* fwht([s .* x; zeros(m - d)]))
  return ProblemInstance(x, y, s)
end

function run_experiment(
  m,
  d,
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
  problem = problem_instance(m, d)
  loss_fn = loss(problem)
  grad_fn = grad(problem)
  z = initializer(problem, 0.5)
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
    CSV.write("pr_large_$(m)_$(d)_polyak.csv", df_polyak)
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
      "pr_large_$(m)_$(d)_bundle.csv",
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
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 1024
  "--run-subgradient"
  help = "Set to run the subgradient method."
  action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["m"],
  args["d"],
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
