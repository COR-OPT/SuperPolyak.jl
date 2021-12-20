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
  problem = SuperPolyak.relu_regression_problem(m, d)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  z_init = SuperPolyak.initializer(problem, δ)
  if run_subgradient
    @info "Running subgradient method..."
    _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
      SuperPolyak.subgradient_method(loss_fn, grad_fn, z_init, ϵ_tol)
    df_polyak = DataFrame(
      t = 1:length(loss_history_polyak),
      fvals = loss_history_polyak,
      cumul_oracle_calls = 0:oracle_calls_polyak,
      cumul_elapsed_time = cumsum(elapsed_time_polyak),
    )
    CSV.write("relu_large_$(m)_$(d)_polyak.csv", df_polyak)
  else
    @info "Running SuperPolyak..."
    result = SuperPolyak.superpolyak(
      loss_fn,
      grad_fn,
      z_init,
      ϵ_decrease = ϵ_decrease,
      ϵ_distance = ϵ_distance,
      ϵ_tol = ϵ_tol,
      η_est = η_est,
      η_lb = η_lb,
      bundle_system_solver = bundle_system_solver,
    )
    df_bundle = save_superpolyak_result(
      "relu_large_$(m)_$(d)_bundle.csv",
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
  default = 500
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 1500
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
