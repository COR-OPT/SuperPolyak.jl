module SuperPolyak

import Arpack: svds
import ElasticArrays: ElasticMatrix
import IterativeSolvers: lsqr!, minres!
import LinearAlgebra
import LinearMaps: LinearMap
import ReverseDiff: GradientTape, gradient!, compile
import SparseArrays: nnz, sparse
import StatsBase: sample

const Diagonal = LinearAlgebra.Diagonal
const Factorization = LinearAlgebra.Factorization
const givens = LinearAlgebra.givens
const lmul! = LinearAlgebra.lmul!
const norm = LinearAlgebra.norm
const normalize = LinearAlgebra.normalize
const normalize! = LinearAlgebra.normalize!
const opnorm = LinearAlgebra.opnorm
const qr = LinearAlgebra.qr
const rank = LinearAlgebra.rank
const UpperTriangular = LinearAlgebra.UpperTriangular

include("problems.jl")
include("sparse_regression_problems.jl")
include("qrinsert.jl")

"""
  polyak_sgm(f::Function, gradf::Function, x₀::Vector{Float64}; ϵ::Float64 = (f(x_0) / 2), min_f::Float64 = 0.0)

Run the Polyak subgradient method until the function value drops below `ϵ`.
Return the final iterate and the total number of calls to the first-order oracle.
"""
function polyak_sgm(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  ϵ::Float64 = (f(x_0) / 2),
  min_f::Float64 = 0.0,
)
  x = x₀[:]
  oracle_calls = 0
  while true
    g = gradf(x)
    x -= (f(x) - min_f) * g / (norm(g)^2)
    oracle_calls += 1
    ((f(x) - min_f) ≤ ϵ) && return x, oracle_calls
  end
end

"""
  polyak_step(f::Function, gradf::Function, x::Vector{Float64}, min_f::Float64 = 0.0)

Run a single polyak step to minimize `f` starting at `x`.
"""
function polyak_step(
  f::Function,
  gradf::Function,
  x::Vector{Float64},
  min_f::Float64 = 0.0,
)
  g = gradf(x)
  return x - (f(x) - min_f) * g / (norm(g)^2)
end

"""
  subgradient_method(f::Function, gradf::Function, x₀::Vector{Float64}; ϵ::Float64 = 1e-15, min_f::Float64 = 0.0)

Run the Polyak subgradient method until the function value drops below `ϵ`.
Return the final iterate, the history of function values along iterates, and
the total number of calls to the first-order oracle.
"""
function subgradient_method(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  ϵ::Float64 = 1e-15,
  min_f::Float64 = 0.0,
)
  x = x₀[:]
  oracle_calls = 0
  fvals = [f(x₀)]
  elapsed_time = [0.0]
  while (fvals[end] > ϵ)
    stats = @timed begin
      g = gradf(x)
      x -= (f(x) - min_f) * g / (norm(g)^2)
    end
    oracle_calls += 1
    push!(fvals, f(x) - min_f)
    push!(elapsed_time, stats.time - stats.gctime)
  end
  return x, fvals, oracle_calls, elapsed_time
end

"""
  fallback_algorithm(f::Function, A::Function, x₀::Vector{Float64},
                     ϵ::Float64 = 1e-15, min_f::Float64 = 0.0;
                     record_loss::Bool = false)

A fallback algorithm to use in the bundle Newton method. Here, `f` is a
callable implementing the loss function, `A` is a mapping that maps `x`
to the next iterate `x₊` and `x₀` is the starting vector.

Terminates when `f(x) - min_f ≤ ϵ` and returns the final iterate, a history
of loss function values (if `record_loss == true`) and the number of oracle
calls to `A`.
"""
function fallback_algorithm(
  f::Function,
  A::Function,
  x₀::Vector{Float64},
  ϵ::Float64 = 1e-15,
  min_f::Float64 = 0.0;
  record_loss::Bool = false,
)
  x = x₀[:]
  oracle_calls = 0
  fvals = [f(x) - min_f]
  elapsed_time = [0.0]
  while (f(x) - min_f) ≥ ϵ
    stats = @timed begin
      x = A(x)
      oracle_calls += 1
      (record_loss) && push!(fvals, f(x) - min_f)
    end
    push!(elapsed_time, stats.time - stats.gctime)
  end
  if (record_loss)
    return x, fvals, oracle_calls, elapsed_time
  else
    return x, oracle_calls, elapsed_time
  end
end

"""
  argmin_parentindex(v::Vector{Float64}, b::BitVector)

Find the argmin of `v` over a subset defined by the bit vector `b` and return
the index in the original vector.
"""
function argmin_parentindex(v::Vector{Float64}, b::BitVector)
  v = view(v, b)
  return first(parentindices(v))[argmin(v)]
end

"""
  build_bundle_lsqr(f::Function, gradf::Function, y₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

An efficient version of the BuildBundle algorithm using the LSQR linear system
solver for the quadratic subproblems.
"""
function build_bundle_lsqr(
  f::Function,
  gradf::Function,
  y₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
  η_est::Float64,
)
  d = length(y₀)
  fvals = zeros(d)
  resid = zeros(d)
  # Resizable bundle matrix. Each column is a bundle element.
  bmtrx = ElasticMatrix(zeros(d, 0))
  y = y₀[:]
  append!(bmtrx, gradf(y))
  fvals[1] = f(y) - min_f + bmtrx[:, 1]' * (y₀ - y)
  y = y₀ - Vector(bmtrx[:, 1])' \ fvals[1]
  resid[1] = f(y) - min_f
  Δ = f(y₀) - min_f
  # Exit early if solution escaped ball.
  if norm(y - y₀) > η * Δ
    return nothing, 1
  end
  # Best solution and function value found so far.
  y_best = y[:]
  f_best = resid[1]
  # Difference dy for the normal equations.
  dy = y - y₀
  for bundle_idx in 2:d
    append!(bmtrx, gradf(y))
    # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
    fvals[bundle_idx] = resid[bundle_idx-1] + bmtrx[:, bundle_idx]' * (y₀ - y)
    At = view(bmtrx, :, 1:bundle_idx)
    lsqr!(
      dy,
      At',
      view(fvals, 1:bundle_idx),
      maxiter = bundle_idx,
      atol = max(Δ, 1e-15),
      btol = max(Δ, 1e-15),
      conlim = 0.0,
    )
    y = y₀ - dy
    resid[bundle_idx] = f(y) - min_f
    # Terminate early if new point escaped ball around y₀.
    if (norm(dy) > η * Δ)
      @debug "Stopping at idx = $(bundle_idx) - reason: diverging"
      return y_best, bundle_idx
    end
    # Terminate early if function value decreased significantly.
    if (Δ < 0.5) && (resid[bundle_idx] < Δ^(1 + η_est))
      return y, bundle_idx
    end
    # Otherwise, update best solution so far.
    if (resid[bundle_idx] < f_best)
      copyto!(y_best, y)
      f_best = resid[bundle_idx]
    end
  end
  return y_best, d
end

"""
  build_bundle_wv(f::Function, gradf::Function, y₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

An efficient version of the BuildBundle algorithm using an incrementally updated
QR algorithm based on the compact WV representation.
"""
function build_bundle_wv(
  f::Function,
  gradf::Function,
  y₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
  η_est::Float64,
)
  d = length(y₀)
  bvect = zeros(d)
  fvals = zeros(d)
  resid = zeros(d)
  y = y₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  copyto!(bvect, gradf(y))
  fvals[1] = f(y) - min_f + bvect' * (y₀ - y)
  # Initialize Q and R
  Q, R = wv_from_vector(bvect)
  y = y₀ - bvect' \ fvals[1]
  resid[1] = f(y) - min_f
  Δ = f(y₀) - min_f
  # Exit early if solution escaped ball.
  if norm(y - y₀) > η * Δ
    return nothing, 1
  end
  # Best solution and function value found so far.
  y_best = y[:]
  f_best = resid[1]
  # Cache right-hand side vector.
  qr_rhs = zero(y)
  for bundle_idx in 2:d
    copyto!(bvect, gradf(y))
    # Invariant: resid[bundle_idx - 1] = f(y) - min_f.
    fvals[bundle_idx] = resid[bundle_idx-1] + bvect' * (y₀ - y)
    # Update the QR decomposition of A' after forming [A' bvect].
    # Q is updated in-place.
    qrinsert_wv!(Q, R, bvect)
    # Terminate early if rank-deficient.
    # size(R) = (d, bundle_idx).
    if R[bundle_idx, bundle_idx] < 1e-15
      @debug "Stopping (idx=$(bundle_idx)) - reason: rank-deficient A"
      y = y₀ - Matrix(Q * R)' \ view(fvals, 1:bundle_idx)
      return (norm(y - y₀) ≤ η * Δ ? y : y_best), bundle_idx
    end
    # Update y by solving the system Q * (inv(R)'fvals).
    # Cost: O(d * bundle_idx)
    Rupper = view(R, 1:bundle_idx, 1:bundle_idx)
    qr_rhs[1:bundle_idx] = Rupper' \ fvals[1:bundle_idx]
    y = y₀ - Q * qr_rhs
    resid[bundle_idx] = f(y) - min_f
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * Δ)
      @debug "Stopping at idx = $(bundle_idx) - reason: diverging"
      return y_best, bundle_idx
    end
    # Terminate early if function value decreased significantly.
    if (Δ < 0.5) && (resid[bundle_idx] < Δ^(1 + η_est))
      return y, bundle_idx
    end
    # Otherwise, update best solution so far.
    if (resid[bundle_idx] < f_best)
      copyto!(y_best, y)
      f_best = resid[bundle_idx]
    end
  end
  return y_best, d
end

"""
  build_bundle_qr(f::Function, gradf::Function, y₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

An efficient version of the BuildBundle algorithm using an incrementally
updated QR factorization. Total runtime is O(d³) instead of O(d⁴).
"""
function build_bundle_qr(
  f::Function,
  gradf::Function,
  y₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
  η_est::Float64,
)
  d = length(y₀)
  bvect = zeros(d)
  fvals = zeros(d)
  resid = zeros(d)
  y = y₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  copyto!(bvect, gradf(y))
  fvals[1] = f(y) - min_f + bvect' * (y₀ - y)
  Q, R = qr(bvect)
  # "Unwrap" LinearAlgebra.QRCompactWYQ type.
  Q = Q * Matrix(1.0 * LinearAlgebra.I, d, d)
  y = y₀ - Q[:, 1] .* (R' \ [fvals[1]])
  resid[1] = f(y) - min_f
  Δ = f(y₀) - min_f
  # Exit early if solution escaped ball.
  if norm(y - y₀) > η * Δ
    return nothing, 1
  end
  # Best solution and function value found so far.
  y_best = y[:]
  f_best = resid[1]
  @debug "bundle_idx = 1 - error: $(resid[1])"
  for bundle_idx in 2:d
    copyto!(bvect, gradf(y))
    fvals[bundle_idx] = f(y) - min_f + bvect' * (y₀ - y)
    # qrinsert!(Q, R, v): QR = Aᵀ and v is the column added.
    # Only assign to R since Q is modified in-place.
    R = qrinsert!(Q, R, bvect)
    # Terminate if rank deficiency is detected.
    if (R[bundle_idx, bundle_idx] < 1e-15)
      @debug "Stopping at idx = $(bundle_idx) - reason: singular R"
      # If R is singular, solve the system from scratch.
      y_new = y₀ - (view(Q, :, 1:bundle_idx) * R)' \ fvals[1:bundle_idx]
      # Return y_new since it will be guaranteed to reduce superlinearly,
      # as long as it does not escape the ball.
      if (norm(y_new - y₀) < η * Δ)
        return y_new, bundle_idx
      else
        return y_best, bundle_idx
      end
    end
    y = y₀ - view(Q, :, 1:bundle_idx) * (R' \ fvals[1:bundle_idx])
    resid[bundle_idx] = f(y) - min_f
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * Δ)
      @debug "Stopping at idx = $(bundle_idx) - reason: diverging"
      return y_best, bundle_idx
    end
    # Terminate early if function value decreased significantly.
    if (Δ < 0.5) && ((f(y) - min_f) < Δ^(1 + η_est))
      return y, bundle_idx
    end
    # Otherwise, update best solution so far.
    if (resid[bundle_idx] < f_best)
      copyto!(y_best, y)
      f_best = resid[bundle_idx]
    end
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  return y_best, d
end

"""
  BundleSystemSolver

An type intended to represent the available implementations of the bundle
system solver.
"""
abstract type BundleSystemSolver end

struct LSQR <: BundleSystemSolver end
struct INCREMENTAL_QR_WV <: BundleSystemSolver end
struct INCREMENTAL_QR_VANILLA <: BundleSystemSolver end

# Choose the appropriate "build_bundle_[...]" method based on the solver.
bundle_func(solver::LSQR) = build_bundle_lsqr
bundle_func(solver::INCREMENTAL_QR_VANILLA) = build_bundle_qr
bundle_func(solver::INCREMENTAL_QR_WV) = build_bundle_wv

"""
  SuperPolyakResult

The result of an invocation of the `superpolyak` algorithm.
"""
struct SuperPolyakResult
  solution::Vector{Float64}
  loss_history::Vector{Float64}
  oracle_calls::Vector{Int}
  elapsed_time::Vector{Float64}
  step_types::Vector{String}
end

"""
  superpolyak(f::Function, gradf::Function, x₀::Vector{Float64};
              ϵ_tol::Float64 = 1e-15, ϵ_decrease::Float64 = (1 / 2),
              ϵ_distance::Float64 = (3 / 2), min_f::Float64 = 0.0,
              bundle_system_solver::BundleSystemSolver = LSQR(),
              η_est::Float64 = 1.0, kwargs...)

Run the SuperPolyak method to find a zero of `f` with subgradient `gradf`,
starting from an initial point `x₀`.

Arguments:
- `f::Function`: A callable implementing the loss function.
- `gradf::Function`: A callable implementing the subgradient mapping.
- `x₀::Vector{Float64}`: The initial iterate of the algorithm.
- `ϵ_tol::Float64 = 1e-15`: The desired tolerance for the solution.
- `ϵ_distance::Float64 = 3/2`: The factor determining diverging iterates. Must be > 1.
- `ϵ_decrease::Float64 = 1/2`: The decrease factor per iteration. Must be < 1.
- `min_f::Float64 = 0.0`: The (known) minimal value of the loss.
- `fallback_alg::Function = polyak_sgm`: A callable that implements the fallback algorithm used.
- `bundle_system_solver::BundleSystemSolver = LSQR()`: The solver to use for the bundle system.
- `η_est::Float64 = 1.0`: An estimate of the (b)-regularity constant.
- `η_lb::Float64 = 0.1`: A lower bound on the estimate of the (b)-regularity constant.

In the above, the product `ϵ_distance * ϵ_decrease` must be smaller than `1`
for the algorithm to converge superlinearly.

Returns:
- `result::SuperPolyakResult`: A struct containing the solution found and a history
  of: loss values, oracle evaluations, elapsed time, and type of step taken per iteration.
"""
function superpolyak(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64};
  ϵ_tol::Float64 = 1e-15,
  ϵ_decrease::Float64 = (1 / 2),
  ϵ_distance::Float64 = (3 / 2),
  min_f::Float64 = 0.0,
  fallback_alg::Function = polyak_sgm,
  bundle_system_solver::BundleSystemSolver = LSQR(),
  η_est::Float64 = 1.0,
  η_lb::Float64 = 0.1,
  kwargs...,
)
  if (ϵ_decrease ≥ 1) || (ϵ_decrease < 0)
    throw(BoundsError(ϵ_decrease, "ϵ_decrease must be between 0 and 1"))
  end
  if (ϵ_decrease * ϵ_distance > 1)
    throw(
      BoundsError(
        ϵ_decrease * ϵ_distance,
        "ϵ_decrease * ϵ_distance must be < 1",
      ),
    )
  end
  x = x₀[:]
  fvals = [f(x₀) - min_f]
  oracle_calls = [0]
  elapsed_time = [0.0]
  step_types = ["NONE"]
  idx = 0
  # Determine the bundle system solver.
  bundle_solver = bundle_func(bundle_system_solver)
  @info "Using bundle system solver: $(typeof(bundle_system_solver))"
  while true
    cumul_time = 0.0
    Δ = fvals[end]
    η = ϵ_distance^(idx)
    bundle_stats = @timed bundle_step, bundle_calls =
      bundle_solver(f, gradf, x, η, min_f, η_est)
    cumul_time += bundle_stats.time - bundle_stats.gctime
    # Adjust η_est if the bundle step did not satisfy the descent condition.
    if !isnothing(bundle_step) &&
       ((f(bundle_step) - min_f) > Δ^(1 + η_est)) &&
       (Δ < 0.5)
      η_est = max(η_est * 0.9, η_lb)
      @debug "Adjusting η_est = $(η_est)"
    end
    if isnothing(bundle_step) || ((f(bundle_step) - min_f) > ϵ_decrease * Δ)
      @debug "Bundle step failed (k=$(idx)) -- using fallback algorithm"
      if (!isnothing(bundle_step)) && ((f(bundle_step) - min_f) < Δ)
        copyto!(x, bundle_step)
      end
      fallback_stats = @timed x, fallback_calls =
        fallback_alg(f, gradf, x, ϵ_decrease * (f(x) - min_f), min_f)
      cumul_time += fallback_stats.time - fallback_stats.gctime
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
      push!(step_types, "FALLBACK")
    else
      @debug "Bundle step successful (k=$(idx))"
      x = bundle_step[:]
      push!(oracle_calls, bundle_calls)
      push!(step_types, "BUNDLE")
    end
    idx += 1
    push!(fvals, f(x) - min_f)
    push!(elapsed_time, cumul_time)
    if fvals[end] ≤ ϵ_tol
      return SuperPolyakResult(x, fvals, oracle_calls, elapsed_time, step_types)
    end
  end
end

end # module
