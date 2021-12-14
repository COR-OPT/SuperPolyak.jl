using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function run_experiment(m, d, k, ϵ_tol, ϵ_decrease, ϵ_distance, η_est, show_amortized)
  problem = SuperPolyak.lasso_problem(m, d, k, 0.1)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = zeros(d)
  τ = 0.9 / (opnorm(problem.A)^2)
  # Define the fallback method.
  proximal_gradient_method(
    loss::Function,
    grad::Function,
    x::Vector{Float64},
    ϵ::Float64,
    min_f::Float64,
  ) = begin
    it = 0
    x₀ = x[:]
    while true
      x₀ = SuperPolyak.proximal_gradient(
        problem.A,
        x₀,
        problem.y,
        problem.λ,
        0.9 / (opnorm(problem.A)^2),
      )
      it += 1
      if loss(x₀) < ϵ
        return x₀, it
      end
    end
  end
  x_bundle, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    x_init[:],
    ϵ_tol = ϵ_tol,
    ϵ_distance = ϵ_distance,
    ϵ_decrease = ϵ_decrease,
    η_est = η_est,
    fallback_alg = proximal_gradient_method,
  )
  cumul_oracle_calls = get_cumul_oracle_calls(oracle_calls, show_amortized)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("lasso_$(m)_$(d)_$(k)_bundle.csv", df_bundle)
  x_vanilla, loss_history_vanilla, oracle_calls_vanilla =
    SuperPolyak.fallback_algorithm(
      loss_fn,
      z -> SuperPolyak.proximal_gradient(problem.A, z, problem.y, problem.λ, τ),
      x_init[:],
      ϵ_tol,
      record_loss = true,
    )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
  )
  CSV.write("lasso_$(m)_$(d)_$(k)_vanilla.csv", df_vanilla)
  semilogy(cumul_oracle_calls, loss_history, "bo--")
  semilogy(loss_history_vanilla, "r--")
  legend(["SuperPolyak", "Proximal Gradient"])
  show()
end

settings = ArgParseSettings()
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The problem dimension."
  default = 100
  "--k"
  arg_type = Int
  help = "The sparsity of the unknown signal."
  default = 1
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 5
  "--eps-tol"
  arg_type = Float64
  help = "The desired tolerance for the final solution."
  default = 1e-12
  "--eps-decrease"
  arg_type = Float64
  help = "The multiplicative decrease factor for the loss."
  default = 0.5
  "--eps-distance"
  arg_type = Float64
  help =
    "A multiplicative factor η for the distance between the initial " *
    "point `y₀` and the output `y` of the bundle Newton method. It " *
    "requires that |y - y₀| < η * f(y₀)."
  default = 1.5
  "--eta-est"
  arg_type = Float64
  help = "The factor of superlinear convergence for early stopping."
  default = 1.5
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
run_experiment(
  args["m"],
  args["d"],
  args["k"],
  args["eps-tol"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eta-est"],
  args["show-amortized"],
)
