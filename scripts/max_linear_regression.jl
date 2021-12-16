using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function run_experiment(
  m,
  d,
  k,
  δ,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  no_amortized,
  plot_inline,
)
  problem = SuperPolyak.max_affine_regression_problem(m, d, k)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  βs_init = SuperPolyak.initializer(problem, δ)
  @info "Running subgradient method..."
  _, loss_history_polyak, oracle_calls_polyak =
    SuperPolyak.subgradient_method(loss_fn, grad_fn, βs_init[:], ϵ_tol)
  df_polyak = DataFrame(
    t = 1:length(loss_history_polyak),
    fvals = loss_history_polyak,
    cumul_oracle_calls = 0:oracle_calls_polyak,
  )
  CSV.write("max_linear_regression_$(m)_$(d)_$(k)_polyak.csv", df_polyak)
  @info "Running SuperPolyak..."
  _, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    βs_init[:],  # Vectorize for compatibility with bundle_newton(...)
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
  )
  cumul_oracle_calls = get_cumul_oracle_calls(oracle_calls, !no_amortized)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("max_linear_regression_$(m)_$(d)_$(k)_bundle.csv", df_bundle)
  if plot_inline
    semilogy(cumul_oracle_calls, loss_history, "bo--")
    semilogy(0:oracle_calls_polyak, loss_history_polyak, "r--")
    legend(["SuperPolyak", "PolyakSGM"])
    show()
  end
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on max-linear regression.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The problem dimension."
  default = 100
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 500
  "--k"
  arg_type = Int
  help = "The number of linear pieces."
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
  args["k"],
  args["initial-distance"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["no-amortized"],
  args["plot-inline"],
)
