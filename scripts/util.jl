using LinearAlgebra
using QPSReader
using SparseArrays

import SuperPolyak

"""
  get_cumul_oracle_calls(oracle_calls::Vector{Int}, show_amortized::Bool)

Return a vector with the cumulative number of oracle calls given a history of
oracle calls per iteration of the algorithm. If `show_amortized` is `true`, the
total number is divided equally among each step.
"""
function get_cumul_oracle_calls(oracle_calls::Vector{Int}, show_amortized::Bool)
  T = length(oracle_calls)
  cumul_oracle_calls =
    show_amortized ? ((0:(T-1)) .* (sum(oracle_calls) ÷ (T - 1))) :
    cumsum(oracle_calls)
  return cumul_oracle_calls
end

"""
  add_base_options(settings::ArgParseSettings)

Add a number of options common to all experiments to the `settings`.
"""
function add_base_options(settings::ArgParseSettings)
  @add_arg_table! settings begin
    "--initial-distance"
      arg_type = Float64
      help = "The normalized initial distance from the solution set."
      default = 1.0
    "--eps-decrease"
      arg_type = Float64
      help = "The multiplicative decrease factor for the loss."
      default = 0.25
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
  read_mps_instance(filename::String, mpsformat=:fixed)

Read an LP instance from an .MPS file.
"""
function read_mps_instance(filename::String, mpsformat = :fixed)
  problem = readqps(filename, mpsformat = mpsformat)
  m, n = problem.ncon, problem.nvar
  A = sparse(problem.arows, problem.acols, problem.avals, m, n)
  equal_lu_ind = @. abs(problem.ucon - problem.lcon) ≤ 1e-15
  finite_l_ind = @. isfinite(problem.lcon) & !equal_lu_ind
  finite_u_ind = @. isfinite(problem.ucon) & !equal_lu_ind
  # Slack variables for constraints Ax + s₁ = u, Ax - s₂ = ℓ.
  nslack_u = sum(finite_u_ind)
  nslack_l = sum(finite_l_ind)
  num_eq = sum(equal_lu_ind)
  # Constraint matrix of the form Ax = b.
  A = [
    A[finite_u_ind, :]
    A[finite_l_ind, :]
    A[equal_lu_ind, :]
  ]
  # [A_{≤} I 0; A_{≥} -I 0; A_{=} 0 0]
  A = [A [
    Diagonal([ones(nslack_u); -ones(nslack_l)])
    zeros(num_eq, nslack_l + nslack_u)
  ]]
  # Right-hand side: [finite(u); finite(l); u_{=}]
  b = [
    problem.ucon[finite_u_ind]
    problem.lcon[finite_l_ind]
    problem.ucon[equal_lu_ind]
  ]
  # All slacks are nonnegative.
  l_var = [problem.lvar; zeros(nslack_l + nslack_u)]
  u_var = [problem.uvar; fill(Inf, nslack_l + nslack_u)]
  return SuperPolyak.LinearProgramTwoSided(
    A,
    b,
    [problem.c; zeros(nslack_l + nslack_u)],
    l_var,
    u_var,
  )
end
