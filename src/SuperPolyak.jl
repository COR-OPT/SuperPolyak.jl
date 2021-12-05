module SuperPolyak

import LinearAlgebra
import SparseArrays
import StatsBase

const Diagonal = LinearAlgebra.Diagonal
const givens = LinearAlgebra.givens
const lmul! = LinearAlgebra.lmul!
const norm = LinearAlgebra.norm
const normalize = LinearAlgebra.normalize
const opnorm = LinearAlgebra.opnorm
const qr = LinearAlgebra.qr
const rank = LinearAlgebra.rank
const sample = StatsBase.sample
const sparse = SparseArrays.sparse
const spzeros = SparseArrays.spzeros

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
  min_f::Float64 = 0.0;
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
  while (fvals[end] > ϵ)
    g = gradf(x)
    x -= (f(x) - min_f) * g / (norm(g)^2)
    oracle_calls += 1
    push!(fvals, f(x) - min_f)
  end
  return x, fvals, oracle_calls
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
  valid_inds = sum((candidates .- y₀) .^2, dims = 1)[:] .< ϵ^2
  (sum(valid_inds) == 0) && return nothing, size(candidates, 2)
  best_idx = argmin_parentindex(residuals, valid_inds)
  @debug "best_idx = $(best_idx) -- R = $(residuals[best_idx])"
  return candidates[:, best_idx], size(candidates, 2)
end

"""
  build_bundle(f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64, min_f::Float64)

Build a bundle for taking a Newton step in the problem of finding zeros of `f`.
Runs `d` steps and returns an element `yi` such that:

  a) `|yᵢ - x₀| < η * (f(x₀) - min_f)`;
  b) f(yᵢ) has the minimal value among all `y` such that `|y - x₀| < η * (f(x₀) - min_f)`.

The algorithm terminates if `yᵢ` satisfies `|yᵢ - x₀| > η * (f(x₀) - min_f)`.

If no iterate satisfies the above, the function outputs `nothing`.
"""
function build_bundle(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
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
  for bundle_idx in 1:d
    bundle[bundle_idx, :] = gradf(y)
    fvals[bundle_idx] = f(y) - min_f
    jvals[bundle_idx] = gradf(y)'y
    A = bundle[1:bundle_idx, :]
    # Compute new point, add to list of candidates
    y = y₀ - A \ ((fvals.-jvals)[1:bundle_idx] + A * y₀)
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * (f(x₀) - min_f))
      @debug "Early stopping at idx = $(bundle_idx)"
      return pick_best_candidate(
        solns[:, 1:(bundle_idx-1)],
        resid[1:(bundle_idx-1)],
        y₀,
        η * (f(x₀) - min_f),
      )
    end
    solns[:, bundle_idx] = y[:]
    resid[bundle_idx] = f(y) - min_f
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  return pick_best_candidate(solns, resid, y₀, η * (f(x₀) - min_f))
end


"""
  build_bundle_qr(f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64, min_f::Float64)

An efficient version of the BuildBundle algorithm using an incrementally
updated QR factorization. Total runtime is O(d³) instead of O(d⁴).
"""
function build_bundle_qr(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  η::Float64,
  min_f::Float64,
)
  d = length(x₀)
  bundle = zeros(d, d)
  fvals = zeros(d)
  jvals = zeros(d)
  resid = zeros(d)
  # Matrix of iterates - column i is the i-th iterate.
  solns = zeros(d, d)
  y₀ = x₀[:]
  y = x₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  bundle[1, :] = gradf(y)
  fvals[1] = f(y) - min_f
  jvals[1] = gradf(y)'y
  Q, R = qr(bundle[1:1, :]')
  # "Unwrap" LinearAlgebra.QRCompactWYQ type.
  Q = Q * Matrix(1.0 * LinearAlgebra.I, d, d)
  y = y₀ - Q[:, 1] .* (R' \ [fvals[1] - jvals[1] + y₀'bundle[1, :]])
  solns[:, 1] = y[:]
  resid[1] = f(y) - min_f
  @debug "bundle_idx = 1 - error: $(resid[1])"
  for bundle_idx in 2:d
    bundle[bundle_idx, :] = gradf(y)
    fvals[bundle_idx] = f(y) - min_f
    jvals[bundle_idx] = gradf(y)'y
    # qrinsert!(Q, R, v): QR = Aᵀ and v is the column added.
    # Only assign to R since Q is modified in-place.
    R = qrinsert!(Q, R, bundle[bundle_idx, :])
    # Terminate if rank deficiency is detected.
    if (rank(R) < bundle_idx)
      @debug "Stopping at idx = $(bundle_idx) - reason: singular R"
      return pick_best_candidate(
        solns[:, 1:(bundle_idx - 1)],
        resid[1:(bundle_idx - 1)],
        y₀,
        η * (f(x₀) - min_f),
      )
    end
    y =
      y₀ -
      view(Q, :, 1:bundle_idx) *
      (R' \ ((fvals.-jvals)[1:bundle_idx] + view(bundle, 1:bundle_idx, :) * y₀))
    # Terminate early if new point escaped ball around y₀.
    if (norm(y - y₀) > η * (f(x₀) - min_f))
      @debug "Stopping at idx = $(bundle_idx) - reason: diverging"
      return pick_best_candidate(
        solns[:, 1:(bundle_idx-1)],
        resid[1:(bundle_idx-1)],
        y₀,
        η * (f(x₀) - min_f),
      )
    end
    solns[:, bundle_idx] = y[:]
    resid[bundle_idx] = f(y) - min_f
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  return pick_best_candidate(solns, resid, y₀, η * (f(x₀) - min_f))
end


"""
  bundle_newton(f::Function, gradf::Function, x₀::Vector{Float64};
                ϵ_tol::Float64 = 1e-15, ϵ_decrease::Float64 = (1 / 2),
                ϵ_distance::Float64 = (3 / 2), min_f::Float64 = 0.0,
                use_qr_bundle::Bool = true, kwargs...)

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
  kwargs...,
)
  if (ϵ_decrease ≥ 1) || (ϵ_decrease < 0)
    throw(BoundsError(ϵ_decrease, "ϵ_decrease must be between 0 and 1"))
  end
  if (ϵ_decrease * ϵ_distance > 1)
    throw(BoundsError(ϵ_decrease * ϵ_distance,
                      "ϵ_decrease * ϵ_distance must be < 1"))
  end
  x = x₀[:]
  fvals = [f(x₀) - min_f]
  oracle_calls = [0]
  idx = 0
  while true
    η = ϵ_distance^(idx)
    bundle_step, bundle_calls =
      (use_qr_bundle) ? build_bundle_qr(f, gradf, x, η, min_f) :
      build_bundle(f, gradf, x, η, min_f)
    if isnothing(bundle_step) || ((f(bundle_step) - min_f) > ϵ_decrease * (f(x) - min_f))
      x, fallback_calls = fallback_alg(f, gradf, x, ϵ_decrease * f(x), min_f)
      # Include the number of oracle calls made by the failed bundle step.
      push!(oracle_calls, fallback_calls + bundle_calls)
    else
      x = bundle_step[:]
      push!(oracle_calls, bundle_calls)
    end
    idx += 1
    push!(fvals, f(x) - min_f)
    (fvals[end] ≤ ϵ_tol) && return x, fvals, oracle_calls
  end
end

end # module
