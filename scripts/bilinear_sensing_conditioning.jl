using ArgParse
using CSV
using DataFrames
using LinearAlgebra
using Printf
using PyPlot
using Random

import SuperPolyak

include("util.jl")


function generate_problem(m::Int, d::Int, r::Int, κ::Int)
  L = randn(m, d)
  R = randn(m, d)
  # Solution with specified condition number.
  W = generate_conditioned_matrix(d, r, sqrt(κ))
  X = generate_conditioned_matrix(d, r, sqrt(κ))
  y = sum((L * W) .* (R * X), dims = 2)[:]
  return SuperPolyak.BilinearSensingProblem(L, R, W, X, y)
end

function run_experiment(
  m,
  d,
  r,
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
  problem = generate_problem(m, d, r, κ)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  z_init = SuperPolyak.initializer(problem, δ)
  @info "Running subgradient method..."
  _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
    SuperPolyak.subgradient_method(loss_fn, grad_fn, z_init[:], ϵ_tol)
  df_polyak = DataFrame(
    t = 1:length(loss_history_polyak),
    fvals = loss_history_polyak,
    cumul_oracle_calls = 0:oracle_calls_polyak,
    cumul_elapsed_time = cumsum(elapsed_time_polyak),
  )
  CSV.write("bilinear_sensing_$(m)_$(d)_$(r)_cond_$(κ)_polyak.csv", df_polyak)
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
    "bilinear_sensing_$(m)_$(d)_$(r)_cond_$(κ)_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo--")
    semilogy(0:oracle_calls_polyak, loss_history_polyak, "r--")
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
    legend(["SuperPolyak", "PolyakSGM"])
    show()
  end
end

settings = ArgParseSettings(
  description = "Compare PolyakSGM with SuperPolyak on bilinear sensing.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The dimension of each column of the unknown signal."
  default = 100
  "--r"
  arg_type = Int
  help = "The number of columns of the unknown signal."
  default = 5
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 1500
  "--condition-number"
  arg_type = Int
  help = "The desired condition number of the ground truth matrix."
  default = 1
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
  args["condition-number"],
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
