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
  plot_inline,
)
  problem = SuperPolyak.phase_retrieval_problem(m, d)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = SuperPolyak.initializer(problem, δ)
  @info "Running subgradient method..."
  _, loss_history_polyak, oracle_calls_polyak, elapsed_time_polyak =
    SuperPolyak.subgradient_method(loss_fn, grad_fn, x_init[:], ϵ_tol)
  df_polyak = DataFrame(
    t = 1:length(loss_history_polyak),
    fvals = loss_history_polyak,
    cumul_oracle_calls = 0:oracle_calls_polyak,
    cumul_elapsed_time = cumsum(elapsed_time_polyak),
  )
  CSV.write("phase_retrieval_$(m)_$(d)_polyak.csv", df_polyak)
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    x_init[:],
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "phase_retrieval_$(m)_$(d)_bundle.csv",
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

settings = ArgParseSettings()
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--d"
  arg_type = Int
  help = "The problem dimension."
  default = 500
  "--m"
  arg_type = Int
  help = "The number of measurements."
  default = 1500
  "--plot-inline"
  help = "Set to plot the results after running the script."
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
  args["plot-inline"],
)
