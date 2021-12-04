using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function run_experiment(m, d, ϵ_tol, show_amortized)
  problem = SuperPolyak.random_linear_program(m, d)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init  = zeros(d + m)
  # Define the fallback method.
  chambolle_pock_method(
    loss::Function,
    grad::Function,
    x₀::Vector{Float64},
    ϵ::Float64,
    min_f::Float64,
  ) = begin
    it = 0
    x = x₀[:]
    while true
      x = SuperPolyak.chambolle_pock(problem, x)
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
    fallback_alg = chambolle_pock_method,
  )
  cumul_oracle_calls = get_cumul_oracle_calls(oracle_calls, show_amortized)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("lp_$(m)_$(d)_bundle.csv", df_bundle)
  _, loss_history_vanilla, oracle_calls_vanilla = SuperPolyak.fallback_algorithm(
    loss_fn,
    z -> SuperPolyak.chambolle_pock(problem, z),
    x_init[:],
    ϵ_tol,
    record_loss = true,
  )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
  )
  CSV.write("lp_$(m)_$(d)_vanilla.csv", df_vanilla)
  semilogy(cumul_oracle_calls, loss_history, "bo--")
  semilogy(0:oracle_calls_vanilla, loss_history_vanilla, "r--")
  legend(["SuperPolyak", "Chambolle-Pock"])
  show()
end

settings = ArgParseSettings()
@add_arg_table! settings begin
  "--m"
    arg_type = Int
    help = "The number of constraints."
    default = 5
  "--d"
    arg_type = Int
    help = "The problem dimension."
    default = 100
  "--eps-tol"
    arg_type = Float64
    help = "The desired tolerance for the final solution."
    default = 1e-12
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
run_experiment(args["m"], args["d"], args["eps-tol"], args["show-amortized"])
