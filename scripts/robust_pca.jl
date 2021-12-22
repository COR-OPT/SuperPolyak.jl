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
  r,
  p,
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
  problem = SuperPolyak.robust_pca_problem(m, d, r, p)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  z_init = SuperPolyak.initializer(problem, δ)
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
      x = SuperPolyak.alternating_projections_step(problem, x)
      it += 1
      if loss(x) < ϵ
        return x, it
      end
    end
  end
  @info "Running alternating projections..."
  _, loss_history_vanilla, oracle_calls_vanilla, elapsed_time_vanilla =
    SuperPolyak.fallback_algorithm(
      loss_fn,
      z -> SuperPolyak.alternating_projections_step(problem, z),
      z_init,
      ϵ_tol,
      record_loss = true,
    )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
    cumul_elapsed_time = cumsum(elapsed_time_vanilla),
  )
  CSV.write("robust_pca_$(m)_$(d)_$(r)_p-$(p)_vanilla.csv", df_vanilla)
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
    fallback_alg = alternating_projections_method,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "robust_pca_$(m)_$(d)_$(r)_p-$(p)_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo--")
    semilogy(0:oracle_calls_vanilla, loss_history_vanilla, "r-")
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
    legend(["SuperPolyak", "Alternating Projections"])
    show()
  end
end

settings = ArgParseSettings()
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--m"
  arg_type = Int
  help = "The number of rows."
  default = 50
  "--d"
  arg_type = Int
  help = "The number of columns."
  default = 500
  "--r"
  arg_type = Int
  help = "The rank of the unknown solution."
  default = 5
  "--p"
  arg_type = Float64
  help = "The probability that an element is corrupted."
  default = 0.05
  range_tester = (x -> 0 ≤ x < 1)
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
  args["p"],
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
