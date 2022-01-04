using ArgParse
using CSV
using DataFrames
using Glob
using LinearAlgebra
using Printf
using PyPlot
using Random

using SuperPolyak

include("util.jl")

function filename_noext(name::String)
  (n, e) = splitext(basename(name))
  while !isempty(e)
    (n, e) = splitext(n)
  end
  return n
end

function run_experiment(
  benchmark_folder,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  bundle_system_solver,
  bundle_budget,
  no_amortized,
)
  for instance in glob(joinpath(benchmark_folder, "*.QPS"))
    solve_instance(
      instance,
      ϵ_decrease,
      ϵ_distance,
      ϵ_tol,
      η_est,
      η_lb,
      bundle_system_solver,
      bundle_budget,
      no_amortized,
    )
  end
end


function solve_instance(
  filename,
  ϵ_decrease,
  ϵ_distance,
  ϵ_tol,
  η_est,
  η_lb,
  bundle_system_solver,
  bundle_budget,
  no_amortized,
)
  problem = SuperPolyak.read_problem_from_qps_file(filename, :fixed)
  stepsize = SuperPolyak.default_stepsize(problem)
  loss_fn = SuperPolyak.chambolle_pock_loss(problem, stepsize)
  grad_fn = SuperPolyak.chambolle_pock_subgradient(problem, stepsize)
  m, n = size(problem.A)
  z_init = zeros(m + n)
  @info "Solving $(filename_noext(filename)) with $m variables and $n constraints."
  @info "rank(P) = $(rank(problem.P))"
  # Define the fallback method.
  fallback_method(
    loss::Function,
    grad::Function,
    x₀::Vector{Float64},
    ϵ::Float64,
    min_f::Float64,
  ) = begin
    it = 0
    x = x₀[:]
    while true
      x = SuperPolyak.chambolle_pock_iteration(problem, x, stepsize)
      it += 1
      if loss(x) < ϵ
        return x, it
      end
    end
  end
  @info "Running Chambolle-Pock..."
  _, loss_history_vanilla, oracle_calls_vanilla, elapsed_time_vanilla =
    SuperPolyak.fallback_algorithm(
      loss_fn,
      z -> SuperPolyak.chambolle_pock_iteration(problem, z, stepsize),
      copy(z_init),
      ϵ_tol,
      record_loss = true,
    )
  df_vanilla = DataFrame(
    t = 1:length(loss_history_vanilla),
    fvals = loss_history_vanilla,
    cumul_oracle_calls = 0:oracle_calls_vanilla,
    cumul_elapsed_time = cumsum(elapsed_time_vanilla)
  )
  CSV.write("qp_$(filename_noext(filename))_vanilla.csv", df_vanilla)
  @info "Running SuperPolyak..."
  result = SuperPolyak.superpolyak(
    loss_fn,
    grad_fn,
    copy(z_init),
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
    ϵ_tol = ϵ_tol,
    η_est = η_est,
    η_lb = η_lb,
    bundle_system_solver = bundle_system_solver,
    fallback_alg = fallback_method,
    bundle_budget = bundle_budget,
  )
  df_bundle = save_superpolyak_result(
    "qp_$(filename_noext(filename))_superpolyak.csv",
    result,
    no_amortized,
  )
end

settings = ArgParseSettings(
  description = "Compare Chambolle-Pock with SuperPolyak on a QP.",
)
settings = add_base_options(settings)
@add_arg_table! settings begin
  "--benchmark-folder"
  help = "Path to a folder containing the instances in .QPS format."
  arg_type = String
  "--bundle-budget"
  help = "The budget for each bundle step."
  arg_type = Int
  default = 50
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(
  args["benchmark-folder"],
  args["eps-decrease"],
  args["eps-distance"],
  args["eps-tol"],
  args["eta-est"],
  args["eta-lb"],
  args["bundle-system-solver"],
  args["bundle-budget"],
  args["no-amortized"],
)
