using LinearAlgebra
using SparseArrays

import SuperPolyak

"""
  get_partial_sums(v::AbstractVector, no_amortized::Bool)

Compute a vector of partial sums of `v`. If `no_amortized` is false, computes
the amortized partial sums, where each increment is identical.
"""
function get_partial_sums(v::AbstractVector, no_amortized::Bool)
  T = length(v)
  # Step should be integer if v has integer type.
  step = eltype(v) == Int ? (sum(v) ÷ (T - 1)) : (sum(v) / (T - 1))
  psum = no_amortized ? cumsum(v) : (0:(T-1)) .* step
  return psum
end

solver_from_string(x) = begin
  if lowercase(x) == "lsqr"
    return SuperPolyak.LSQR()
  elseif lowercase(x) == "incremental_qr_wv"
    return SuperPolyak.INCREMENTAL_QR_WV()
  elseif lowercase(x) == "incremental_qr_vanilla"
    return SuperPolyak.INCREMENTAL_QR_VANILLA()
  else
    throw(ErrorException("Unknown bundle system solver $(x)"))
  end
end

# Custom parser for bundle system solver type.
function ArgParse.parse_item(
  ::Type{SuperPolyak.BundleSystemSolver},
  input::AbstractString,
)
  return solver_from_string(input)
end

"""
  add_base_options(settings::ArgParseSettings)

Add a number of options common to all experiments to the `settings`.
"""
function add_base_options(settings::ArgParseSettings)
  @add_arg_table! settings begin
    "--bundle-system-solver"
    arg_type = SuperPolyak.BundleSystemSolver
    help =
      "The bundle system solver to use. One of: LSQR, INCREMENTAL_QR_WV, " *
      "and INCREMENTAL_QR_VANILLA."
    default = SuperPolyak.INCREMENTAL_QR_WV()
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
    help =
      "A multiplicative factor η for the distance between the initial " *
      "point `y₀` and the output `y` of the bundle Newton method. It " *
      "requires that |y - y₀| < η * f(y₀)."
    default = 1.5
    "--eps-tol"
    arg_type = Float64
    help = "Terminate if the loss function drops below this tolerance."
    default = 1e-14
    "--eta-est"
    arg_type = Float64
    help = "The initial estimate of the (b)-regularity constant."
    default = 1.0
    "--eta-lb"
    arg_type = Float64
    help = "A lower bound for the estimated (b)-regularity constant."
    default = 0.1
    "--seed"
    arg_type = Int
    help = "The seed for the random number generator."
    default = 999
    "--no-amortized"
    help = "Disable amortization of cumulative oracle evaluations."
    action = :store_true
  end
  return settings
end

"""
  save_superpolyak_result(name::String, result::SuperPolyak.SuperPolyakResult,
                          no_amortized::Bool)

Save the results of a run of `superpolyak` to a .CSV file under `name`.
"""
function save_superpolyak_result(
  name::String,
  result::SuperPolyak.SuperPolyakResult,
  no_amortized::Bool,
)
  cumul_oracle_calls = get_partial_sums(result.oracle_calls, no_amortized)
  cumul_elapsed_time = get_partial_sums(result.elapsed_time, no_amortized)
  df_bundle = DataFrame(
    t = 1:length(result.loss_history),
    fvals = result.loss_history,
    cumul_oracle_calls = cumul_oracle_calls,
    cumul_elapsed_time = cumul_elapsed_time,
    step_type = result.step_types,
  )
  CSV.write(name, df_bundle)
  return df_bundle
end

"""
  generate_conditioned_matrix(d::Int, r::Int, κ::Float64)

Generate a `d × r` with condition number κ and unit Frobenius
norm.
"""
function generate_conditioned_matrix(d::Int, r::Int, κ::Float64)
  X = Matrix(qr(randn(d, r)).Q)
  Σ = Diagonal(range(1, κ, length = r))
  return (X * Σ) / norm(Σ)
end
