module SuperPolyak

import ElasticArrays: ElasticMatrix
import IterativeSolvers: lsqr!, minres!
import LinearAlgebra
import LinearMaps: LinearMap
import ReverseDiff: GradientTape, gradient!, compile
import SparseArrays
import StatsBase

const Diagonal = LinearAlgebra.Diagonal
const givens = LinearAlgebra.givens
const lmul! = LinearAlgebra.lmul!
const norm = LinearAlgebra.norm
const normalize = LinearAlgebra.normalize
const normalize! = LinearAlgebra.normalize!
const opnorm = LinearAlgebra.opnorm
const qr = LinearAlgebra.qr
const rank = LinearAlgebra.rank
const sample = StatsBase.sample
const sparse = SparseArrays.sparse
const SparseMatrixCSC = SparseArrays.SparseMatrixCSC
const spzeros = SparseArrays.spzeros
const UpperTriangular = LinearAlgebra.UpperTriangular

# An abstract type encoding an optimization problem. All concrete problem
# instances should be subtypes of `OptProblem`.
abstract type OptProblem end

include("chambolle_pock.jl")
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
  while (f(x) - min_f) ≥ ϵ
    x = A(x)
    oracle_calls += 1
    (record_loss) && push!(fvals, f(x) - min_f)
  end
  if (record_loss)
    return x, fvals, oracle_calls
  else
    return x, oracle_calls
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
  pick_best_candidate(candidates, residuals, y₀, ϵ)

Pick the best candidate among a list of `candidates` (with one candidate
per column) with corresponding `residuals`. Return the candidate with the
lowest residual among all candidates `y` satisfying `|y - y₀| < ϵ` (or `nothing`
if no such candidate exists) as well as the number of candidates considered.
"""
function pick_best_candidate(candidates, residuals, y₀, ϵ)
  valid_inds = sum((candidates .- y₀) .^ 2, dims = 1)[:] .< ϵ^2
  (sum(valid_inds) == 0) && return nothing, size(candidates, 2)
  best_idx = argmin_parentindex(residuals, valid_inds)
  @debug "best_idx = $(best_idx) -- R = $(residuals[best_idx])"
  return candidates[:, best_idx], size(candidates, 2)
end

"""
  build_bundle(f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

Build a bundle for taking a Newton step in the problem of finding zeros of `f`.
Runs `d` steps and returns an element `yi` such that:

  a) `|yᵢ - x₀| < η * (f(x₀) - min_f)`;
  b) f(yᵢ) has the minimal value among all `y` such that `|y - x₀| < η * (f(x₀) - min_f)`.

The algorithm terminates if `yᵢ` satisfies `|yᵢ - x₀| > η * (f(x₀) - min_f)` or
`f(yᵢ) - min_f < (f(x₀) - min_f)^(1+η_est)`.

If no iterate satisfies the above, the function outputs `nothing`.
"""
function build_bundle(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
  η_est::Float64,
)
  d = length(x₀)
  # Allocate bundle, function + jacobian values.
  bundle = zeros(d, d)
  fvals = zeros(d)
  jvals = zeros(d)
  resid = zeros(d)
  # Matrix of iterates - column i is the i-th iterate.
  solns = zeros(d, d)
  y₀ = x₀[:]
  y = x₀[:]
  Δ = f(x₀) - min_f
  for bundle_idx in 1:d
    bundle[bundle_idx, :] = gradf(y)
    fvals[bundle_idx] = f(y) - min_f
    jvals[bundle_idx] = gradf(y)' * (y - y₀)
    A = view(bundle, 1:bundle_idx, :)
    # Compute new point, add to list of candidates
    y = y₀ - A \ (fvals[1:bundle_idx] - jvals[1:bundle_idx])
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * Δ)
      @debug "Early stopping at idx = $(bundle_idx)"
      return pick_best_candidate(
        solns[:, 1:(bundle_idx-1)],
        resid[1:(bundle_idx-1)],
        y₀,
        η * Δ,
      )
    end
    # Terminate early if function value decreased significantly.
    if (Δ < 0.5) && ((f(y) - min_f) < Δ^(1 + η_est))
      return y, bundle_idx
    end
    solns[:, bundle_idx] = y[:]
    resid[bundle_idx] = f(y) - min_f
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  return pick_best_candidate(solns, resid, y₀, η * Δ)
end

"""
  build_bundle_lsqr(f::Function, gradf::Function, y₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

An efficient version of the BuildBundle algorithm using a recycled Krylov method.
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
  bvect = zeros(d)
  fvals = zeros(d)
  resid = zeros(d)
  bmtrx = zeros(d, d)
  y = y₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  bmtrx[:, 1] = gradf(y)
  fvals[1] = f(y) - min_f + bmtrx[:, 1]' * (y₀ - y)
  y = y₀ - bmtrx[:, 1]' \ fvals[1]
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
  dy = zeros(d)
  for bundle_idx in 2:d
    bmtrx[:, bundle_idx] = gradf(y)
    fvals[bundle_idx] = f(y) - min_f + bmtrx[:, bundle_idx]' * (y₀ - y)
    At = view(bmtrx, :, 1:bundle_idx)
    lsqr!(dy, At', fvals[1:bundle_idx], maxiter=bundle_idx,
          atol = Δ, btol = Δ, conlim = 0.0)
    y = y₀ - dy
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
  end
  return y_best, d
end

"""
  build_bundle_wv(f::Function, gradf::Function, y₀::Vector{Float64}, η::Float64, min_f::Float64, η_est::Float64)

An efficient version of the BuildBundle algorithm using an incrementally updated
QR algorithm.
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
  for bundle_idx in 2:d
    copyto!(bvect, gradf(y))
    fvals[bundle_idx] = f(y) - min_f + bvect' * (y₀ - y)
    # Update the QR decomposition of A' after forming [A' bvect].
    # Q is updated in-place.
    qrinsert_wv!(Q, R, bvect)
    # Terminate early if rank-deficient.
    # size(R) = (d, bundle_idx).
    if R[bundle_idx, bundle_idx] < 1e-15
      @debug "Stopping (idx=$(bundle_idx)) - reason: diverging"
      y =
        y₀ - (Q * R)' \ view(fvals, 1:bundle_idx)
      return (norm(y - y₀) ≤ η * Δ ? y : y_best), bundle_idx
    end
    # Update y by solving the system Q * (inv(R)'fvals).
    # Cost: O(d * bundle_idx)
    Rupper = view(R, 1:bundle_idx, 1:bundle_idx)
    y = y₀ - Q * [(Rupper' \ view(fvals, 1:bundle_idx)); zeros(d - bundle_idx)]
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
      y_new = y₀ - (Q * R)' \ fvals[1:bundle_idx]
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
  bundle_newton(f::Function, gradf::Function, x₀::Vector{Float64};
                ϵ_tol::Float64 = 1e-15, ϵ_decrease::Float64 = (1 / 2),
                ϵ_distance::Float64 = (3 / 2), min_f::Float64 = 0.0,
                use_qr_bundle::Bool = true, η_est::Float64 = 2.0, kwargs...)

Run the bundle Newton method to find a zero of `f` with Jacobian `gradf`,
starting from an initial point `x₀`.
"""
function bundle_newton(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64};
  ϵ_tol::Float64 = 1e-15,
  ϵ_decrease::Float64 = (1 / 2),
  ϵ_distance::Float64 = (3 / 2),
  min_f::Float64 = 0.0,
  fallback_alg::Function = polyak_sgm,
  use_qr_bundle::Bool = true,
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
  idx = 0
  while true
    cumul_time = 0.0
    η = ϵ_distance^(idx)
    bundle_stats = @timed bundle_step, bundle_calls =
      (use_qr_bundle) ? build_bundle_qr(f, gradf, x, η, min_f, η_est) :
      build_bundle_wv(f, gradf, x, η, min_f, η_est)
    cumul_time += bundle_stats.time - bundle_stats.gctime
    # Adjust η_est if the bundle step did not satisfy the descent condition.
    if !isnothing(bundle_step) &&
       ((f(bundle_step) - min_f) > (f(x) - min_f)^(1 + η_est)) &&
       ((f(x) - min_f) < 0.5)
      η_est = max(η_est * 0.9, η_lb)
      @debug "Adjusting η_est = $(η_est)"
    end
    if isnothing(bundle_step) ||
       ((f(bundle_step) - min_f) > ϵ_decrease * (f(x) - min_f))
      @debug "Bundle step failed (k=$(idx)) -- using fallback algorithm"
      fallback_stats = @timed x, fallback_calls =
        fallback_alg(f, gradf, x, ϵ_decrease * f(x), min_f)
      cumul_time += fallback_stats.time - fallback_stats.gctime
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
    else
      @debug "Bundle step successful (k=$(idx))"
      x = bundle_step[:]
      push!(oracle_calls, bundle_calls)
    end
    idx += 1
    push!(fvals, f(x) - min_f)
    push!(elapsed_time, cumul_time)
    (fvals[end] ≤ ϵ_tol) && return x, fvals, oracle_calls, elapsed_time
  end
end

end # module
