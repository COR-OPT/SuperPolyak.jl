using ArgParse
using CSV
using DataFrames
using Printf
using PyPlot
using Random

using SuperPolyak

function run_experiment(m, d, δ, ϵ_decrease, ϵ_distance, show_amortized)
  problem = SuperPolyak.phase_retrieval_problem(m, d)
  loss_fn = SuperPolyak.loss(problem)
  grad_fn = SuperPolyak.subgradient(problem)
  x_init = SuperPolyak.initializer(problem, δ)
  _, loss_history, oracle_calls = SuperPolyak.bundle_newton(
    loss_fn,
    grad_fn,
    x_init,
    ϵ_decrease = ϵ_decrease,
    ϵ_distance = ϵ_distance,
  )
  T = length(oracle_calls)
  cumul_oracle_calls =
    show_amortized ? ((1:T) .* (sum(oracle_calls) ÷ T)) : cumsum(oracle_calls)
  df_bundle = DataFrame(
    t = 1:length(loss_history),
    fvals = loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
  )
  CSV.write("phase_retrieval_$(m)_$(d).csv", df_bundle)
  _, loss_history_polyak, oracle_calls_polyak = SuperPolyak.subgradient_method(
    loss_fn,
    grad_fn,
    x_init,
  )
  semilogy(cumul_oracle_calls, loss_history, "bo--")
  semilogy(0:oracle_calls_polyak, loss_history_polyak, "r--")
  legend(["BundleNewton", "PolyakSGM"])
  show()
end

settings = ArgParseSettings()
@add_arg_table! settings begin
  "--d"
    arg_type = Int
    help = "The problem dimension."
    default = 100
  "--m"
    arg_type = Int
    help = "The number of measurements."
    default = 500
  "--initial-distance"
    arg_type = Float64
    help = "The normalized initial distance from the solution set."
    default = 1.0
  "--eps-decrease"
    arg_type = Float64
    help = "The multiplicative decrease factor for the loss."
    default = 0.5
  "--eps-distance"
    arg_type = Float64
    help = "A multiplicative factor η for the distance between the initial " *
           "point `y₀` and the output `y` of the bundle Newton method. It " *
           "requires that |y - y₀| < η * f(y₀)."
    default = 1.5
  "--seed"
    arg_type = Int
    help = "The seed for the random number generator."
    default = 123
  "--show-amortized"
    help = "Set to plot the residual vs. the amortized number of oracle calls."
    action = :store_true
end

args = parse_args(settings)
Random.seed!(args["seed"])
run_experiment(args["m"], args["d"], args["initial-distance"],
               args["eps-decrease"], args["eps-distance"],
               args["show-amortized"])
