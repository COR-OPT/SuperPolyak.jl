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
    show_amortized ? ((0:(T-1)) .* (sum(oracle_calls) ÷ (T-1))) : cumsum(oracle_calls)
  return cumul_oracle_calls
end

"""
  read_mps_instance(filename::String, mpsformat=:fixed)

Read an LP instance from an .MPS file.
"""
function read_mps_instance(filename::String, mpsformat=:fixed)
  problem = readqps(filename, mpsformat=mpsformat)
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
  A = [A[finite_u_ind, :];
       A[finite_l_ind, :];
       A[equal_lu_ind, :]]
  # [A_{≤} I 0; A_{≥} -I 0; A_{=} 0 0]
  A = [A [Diagonal([ones(nslack_u); -ones(nslack_l)]);
          zeros(num_eq, nslack_l + nslack_u)]]
  # Right-hand side: [finite(u); finite(l); u_{=}]
  b = [problem.ucon[finite_u_ind];
       problem.lcon[finite_l_ind];
       problem.ucon[equal_lu_ind]]
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
