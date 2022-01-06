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
  problem = SuperPolyak.phase_retrieval_problem(m, d, elt_type = ComplexF64)
  loss_fn = SuperPolyak.loss_altproj(problem)
  grad_fn = SuperPolyak.subgradient_altproj(problem)
  x_init = SuperPolyak.initializer_altproj(problem, δ)
  # Fallback method.
  step_fn = SuperPolyak.alternating_projections_step(problem)
  alternating_projections_method(
    loss::Function,
    grad::Function,
    x₀::AbstractVector,
    ϵ::Float64,
    min_f::Float64,
  ) = begin
    it = 0
    x = x₀[:]
    while true
      x = step_fn(x)
      it += 1
      if loss(x) ≤ ϵ
        return x, it
      end
    end
  end
  @info "Running alternating projections method..."
  _, loss_history_altproj, oracle_calls_altproj, elapsed_time_altproj =
    SuperPolyak.fallback_algorithm(
      loss_fn,
      step_fn,
      copy(x_init),
      ϵ_tol,
      record_loss = true,
    )
  df_altproj = DataFrame(
    t = 1:length(loss_history_altproj),
    fvals = loss_history_altproj,
    cumul_oracle_calls = 0:oracle_calls_altproj,
    cumul_elapsed_time = cumsum(elapsed_time_altproj),
  )
  CSV.write("complex_phase_retrieval_$(m)_$(d)_altproj.csv", df_altproj)
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    copy(x_init),
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    bundle_system_solver = bundle_system_solver,
    fallback_alg = alternating_projections_method,
  )
  df_bundle = save_superpolyak_result(
    "complex_phase_retrieval_$(m)_$(d)_bundle.csv",
    result,
    no_amortized,
  )
  if plot_inline
    semilogy(df_bundle.cumul_oracle_calls, df_bundle.fvals, "bo--")
    semilogy(0:oracle_calls_altproj, loss_history_altproj, "r--")
    xlabel("Oracle calls")
    ylabel(L"$ f(x_k) - f^* $")
    legend(["SuperPolyak", "Alt. Projections"])
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
