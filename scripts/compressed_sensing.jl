using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function run_experiment(m, d, k, δ, ϵ_tol, show_amortized)
  problem = SuperPolyak.compressed_sensing_problem(m, d, k)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = SuperPolyak.initializer(problem, δ)
  # Define the fallback method.
  alternating_projections_method(
    loss::Function,
    grad::Function,
    x₀::Vector{Float64},
    ϵ::Float64,
    min_f::Float64,
  ) = begin
    it = 0
    x = x₀[:]
    while true
      x = SuperPolyak.proj_sparse(
        SuperPolyak.proj_range(problem.A, x, problem.y),
        k,
      )
      it += 1
      if loss(x) < ϵ
        return x, it
      end
    end
  end
  _, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    x_init[:],
    ϵ_tol = ϵ_tol,
    fallback_alg = alternating_projections_method,
  )
  cumul_oracle_calls = get_cumul_oracle_calls(oracle_calls, show_amortized)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("compressed_sensing_$(m)_$(d)_$(k)_bundle.csv", df_bundle)
  _, loss_history_vanilla, oracle_calls_vanilla = SuperPolyak.fallback_algorithm(
    loss_fn,
    z -> SuperPolyak.proj_sparse(
      SuperPolyak.proj_range(problem.A, z, problem.y),
      k,
    ),
    x_init[:],
    ϵ_tol,
    record_loss = true,
  )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
  )
  CSV.write("compressed_sensing_$(m)_$(d)_$(k)_vanilla.csv", df_vanilla)
  semilogy(cumul_oracle_calls, loss_history, "bo--")
  semilogy(0:oracle_calls_vanilla, loss_history_vanilla, "r-")
  legend(["SuperPolyak", "Alternating Projections"])
  show()
end

settings = ArgParseSettings()
@add_arg_table! settings begin
  "--m"
    arg_type = Int
    help = "The number of measurements."
    default = 50
  "--d"
    arg_type = Int
    help = "The problem dimension."
    default = 500
  "--k"
    arg_type = Int
    help = "The sparsity of the unknown solution."
    default = 5
  "--initial-distance"
    arg_type = Float64
    help = "The normalized initial distance from the solution set."
    default = 0.5
  "--eps-tol"
    arg_type = Float64
    help = "The desired tolerance for the final solution."
    default = 1e-15
  "--seed"
    arg_type = Int
    help = "The seed for the random number generator."
    default = 999
  "--show-amortized"
    help = "Set to plot the residual vs. the amortized number of oracle calls."
    action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(args["m"], args["d"], args["k"], args["initial-distance"],
               args["eps-tol"], args["show-amortized"])
