module SuperPolyak

import LinearAlgebra
import StatsBase

const givens = LinearAlgebra.givens
const lmul! = LinearAlgebra.lmul!
const norm = LinearAlgebra.norm
const normalize = LinearAlgebra.normalize
const qr = LinearAlgebra.qr
const sample = StatsBase.sample

include("problems.jl")
include("qrinsert.jl")

"""
  polyak_sgm(f::Function, gradf::Function, x₀::Vector{Float64}; ϵ::Float64 = (f(x_0) / 2), min_f::Float64 = 0.0)

Run the Polyak subgradient method assuming a minimal objective value of `0`
until the function value drops below `ϵ`.
"""
function polyak_sgm(
  f::Function,
  gradf::Function,
  x₀::Vector{Float64},
  ϵ::Float64 = (f(x_0) / 2),
  min_f::Float64 = 0.0,
)
  x = x₀[:]
  while true
    g = gradf(x)
    x -= (f(x) - min_f) * g / (norm(g)^2)
    (f(x) ≤ ϵ) && return x
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
  build_bundle(f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64)

Build a bundle for taking a Newton step in the problem of finding zeros of `f`.
Runs `d` steps and returns an element `yi` such that:

  a) `|yi - x₀| < η * f(x₀)`;
  b) f(yi) has the minimal value among all `y` such that `|y - x₀| < η * f(x₀)`.

If no iterate satisfies the above, the function outputs `nothing`.
"""
function build_bundle(
  f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64,
)
  d = length(x₀)
  # Allocate bundle, function + jacobian values.
  bundle = zeros(d, d)
  fvals  = zeros(d)
  jvals  = zeros(d)
  resid  = zeros(d)
  # Matrix of iterates - column i is the i-th iterate.
  solns  = zeros(d, d)
  y₀ = x₀[:]
  y = x₀[:]
  for bundle_idx in 1:d
    bundle[bundle_idx, :] = gradf(y)
    fvals[bundle_idx] = f(y)
    jvals[bundle_idx] = gradf(y)'y
    A = bundle[1:bundle_idx, :]
    # Compute new point, add to list of candidates
    y = y₀ - A \ ((fvals .- jvals)[1:bundle_idx] + A * y₀)
    solns[:, bundle_idx] = y[:]
    resid[bundle_idx] = f(y)
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  # Index i is valid if `|yᵢ - y₀| < η * f(x₀)`
  valid_inds = sum((solns .- y₀).^2, dims=1)[:] .< (η * f(x₀))^2
  (sum(valid_inds) == 0) && return nothing
  best_idx = argmin_parentindex(resid, valid_inds)
  @debug "best_idx = $(best_idx) -- R = $(resid[best_idx])"
  return solns[:, best_idx]
end


"""
  build_bundle_qr(f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64)

An efficient version of the BuildBundle algorithm using an incrementally
updated QR factorization. Total runtime is O(d³) instead of O(d⁴).
"""
function build_bundle_qr(
  f::Function, gradf::Function, x₀::Vector{Float64}, η::Float64,
)
  d = length(x₀)
  bundle = zeros(d, d)
  fvals  = zeros(d)
  jvals  = zeros(d)
  resid  = zeros(d)
  # Matrix of iterates - column i is the i-th iterate.
  solns = zeros(d, d)
  y₀ = x₀[:]
  y  = x₀[:]
  # To obtain a solution equivalent to applying the pseudoinverse, we use the
  # QR factorization of the transpose of the bundle matrix. This is because the
  # minimum-norm solution to Ax = b when A is full row rank can be found via the
  # QR factorization of Aᵀ.
  bundle[1, :] = gradf(y)
  fvals[1] = f(y)
  jvals[1] = gradf(y)'y
  Q, R = qr(bundle[1:1, :]')
  # "Unwrap" LinearAlgebra.QRCompactWYQ type.
  Q = Q * Matrix(1.0 * LinearAlgebra.I, d, d)
  y = y₀ - Q[:, 1] .* (R' \ [fvals[1] - jvals[1] + y₀'bundle[1, :]])
  solns[:, 1] = y[:]
  resid[1] = f(y)
  @debug "bundle_idx = 1 - error: $(resid[1])"
  for bundle_idx in 2:d
    bundle[bundle_idx, :] = gradf(y)
    fvals[bundle_idx] = f(y)
    jvals[bundle_idx] = gradf(y)'y
    # qrinsert!(Q, R, v): QR = Aᵀ and v is the column added.
    # Only assign to R since Q is modified in-place.
    R = qrinsert!(Q, R, bundle[bundle_idx, :])
    # TODO: Terminate if rank deficiency is detected.
    y = y₀ - view(Q, :, 1:bundle_idx) * (
      R' \ ((fvals .- jvals)[1:bundle_idx] + view(bundle, 1:bundle_idx, :) * y₀)
    )
    solns[:, bundle_idx] = y[:]
    resid[bundle_idx] = f(y)
    @debug "bundle_idx = $(bundle_idx) - error: $(resid[bundle_idx])"
  end
  # Index `i` is valid if `|yᵢ - y₀| < η * f(x₀)`.
  valid_inds = sum((solns .- y₀).^2, dims=1)[:] .< (η * f(x₀))^2
  (sum(valid_inds) == 0) && return nothing
  best_idx = argmin_parentindex(resid, valid_inds)
  @debug "best_idx = $(best_idx) -- error = $(resid[best_idx])"
  return solns[:, best_idx]
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
  use_qr_bundle::Bool = true,
  kwargs...
)
  # TODO: Check input for values of `ϵ_decrease`, `ϵ_distance`
  x = x₀[:]
  fvals = [f(x₀)]
  idx = 0
  while true
    η = ϵ_distance^(idx)
    bundle_step = (use_qr_bundle) ? build_bundle_qr(f, gradf, x, η) : build_bundle(f, gradf, x, η)
    if isnothing(bundle_step) || (f(bundle_step) > ϵ_decrease * f(x))
      x = polyak_sgm(f, gradf, x, ϵ_decrease * f(x), min_f)
    else
      x = bundle_step[:]
    end
    idx += 1
    push!(fvals, f(x))
    (fvals[end] ≤ ϵ_tol) && return x, fvals
  end
end

end # module
