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
  bundle_system_solver,
  no_amortized,
  plot_inline,
)
  problem = SuperPolyak.lasso_problem(m, d, r, 0.1)
  # Use same τ for loss and fallback method
  τ = 0.95 / (opnorm(problem.A)^2)
  loss_fn = SuperPolyak.loss(problem, τ)
  grad_fn = SuperPolyak.subgradient(problem, τ)
  x_init = zeros(d)
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
  x_vanilla, loss_history_vanilla, oracle_calls_vanilla, elapsed_time_vanilla =
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
    cumul_elapsed_time = cumsum(elapsed_time_vanilla),
  )
  CSV.write("lasso_$(m)_$(d)_$(r)_vanilla.csv", df_vanilla)
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    x_init[:],
    ϵ_distance = ϵ_distance,
    ϵ_decrease = ϵ_decrease,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    fallback_alg = proximal_gradient_method,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "lasso_$(m)_$(d)_$(r)_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo--")
    semilogy(loss_history_vanilla, "r--")
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
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
  args["bundle-system-solver"],
  args["no-amortized"],
  args["plot-inline"],
)
