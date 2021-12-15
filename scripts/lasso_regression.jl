using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function run_experiment(
  m,
  d,
  r,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  no_amortized,
  plot_inline,
)
  problem = SuperPolyak.lasso_problem(m, d, r, 0.1)
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
      x₀ = SuperPolyak.proximal_gradient(problem, x₀, τ)
      it += 1
      if loss(x₀) < ϵ
        return x₀, it
      end
    end
  end
  @info "Running proximal gradient method..."
  x_vanilla, loss_history_vanilla, oracle_calls_vanilla =
    SuperPolyak.fallback_algorithm(
      loss_fn,
      z -> SuperPolyak.proximal_gradient(problem, z, τ),
      x_init[:],
      ϵ_tol,
      record_loss = true,
    )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
  )
  CSV.write("lasso_$(m)_$(d)_$(r)_vanilla.csv", df_vanilla)
  @info "Running SuperPolyak..."
  x_bundle, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    x_init[:],
    ϵ_distance = ϵ_distance,
    ϵ_decrease = ϵ_decrease,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    fallback_alg = proximal_gradient_method,
  )
  cumul_oracle_calls = get_cumul_oracle_calls(oracle_calls, !no_amortized)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("lasso_$(m)_$(d)_$(r)_bundle.csv", df_bundle)
  if plot_inline
    semilogy(cumul_oracle_calls, loss_history, "bo--")
    semilogy(loss_history_vanilla, "r--")
    legend(["SuperPolyak", "Proximal Gradient"])
    show()
  end
end

settings = ArgParseSettings(
  description = "Compare proximal gradient and SuperPolyak for LASSO regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The problem dimension."
  default = 100
  "--r"
  arg_type = Int
  help = "The sparsity of the unknown signal."
  default = 1
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 5
  "--plot-inline"
  help = "Set to plot the results after running the script."
  action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["m"],
  args["d"],
  args["r"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["no-amortized"],
  args["plot-inline"],
)
