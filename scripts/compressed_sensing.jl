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
  bundle_system_solver,
  no_amortized,
  plot_inline,
)
  problem = SuperPolyak.compressed_sensing_problem(m, d, k)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = SuperPolyak.initializer(problem, δ)
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
      x = SuperPolyak.proj_sparse(
        SuperPolyak.proj_range(problem, x),
        k,
      )
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
      z -> SuperPolyak.proj_sparse(
        SuperPolyak.proj_range(problem, z),
        k,
      ),
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
  CSV.write("compressed_sensing_$(m)_$(d)_$(k)_vanilla.csv", df_vanilla)
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
    fallback_alg = alternating_projections_method,
    bundle_system_solver = bundle_system_solver,
  )
  df_bundle = save_superpolyak_result(
    "compressed_sensing_$(m)_$(d)_$(k)_bundle.csv",
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
  help = "The number of measurements."
  default = 50
  "--d"
  arg_type = Int
  help = "The problem dimension."
  default = 500
  "--k"
  arg_type = Int
  help = "The sparsity of the unknown solution."
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
  args["bundle-system-solver"],
  args["no-amortized"],
  args["plot-inline"],
)
