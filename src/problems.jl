"""
  problems.jl: Implementations of standard problems from machine learning and
  signal processing for the bundle Newton framework.
"""

struct PhaseRetrievalProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
end

function loss(problem::PhaseRetrievalProblem)
  return z -> (1 / length(problem.y)) * norm((problem.A * z).^2 .- problem.y, 1)
end

function subgradient(problem::PhaseRetrievalProblem)
  m = length(problem.y)
  A = problem.A
  y = problem.y
  return z -> (2 / m) * A' * (sign.((A * z).^2 - y) .* (A * z))
end

"""
  initializer(problem::PhaseRetrievalProblem, δ::Float64)

Return an initial estimate `x₀` that is `δ`-close to the ground truth in
normalized distance.
"""
function initializer(problem::PhaseRetrievalProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end

function phase_retrieval_problem(m, d)
  A = randn(m, d)
  x = normalize(randn(d))
  return PhaseRetrievalProblem(A, x, (A * x).^2)
end

struct BilinearSensingProblem
  L::Matrix{Float64}
  R::Matrix{Float64}
  w::Vector{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
end

function loss(problem::BilinearSensingProblem)
  L, R = problem.L, problem.R
  y = problem.y
  m = length(y)
  d = length(problem.w)
  return z -> (1 / m) * norm((L * z[1:d]) .* (R * z[(d+1):end]) .- y, 1)
end

function subgradient(problem::BilinearSensingProblem)
  L, R = problem.L, problem.R
  y = problem.y
  m = length(y)
  d = length(problem.w)
  g(z) = begin
    w = z[1:d]
    x = z[(d+1):end]
    r = (L * w) .* (R * x) .- y
    s = sign.(r)
    return (1 / m) .* vcat(L' * (s .* (R * x)),
                           R' * (s .* (L * w)))
  end
  return g
end

"""
  initializer(problem::BilinearSensingProblem, δ::Float64)

Return an initial estimate `(w₀, x₀)` that is `δ`-close to the ground truth
in normalized distance. Output a single vector containing `w₀` and `x₀` stacked
vertically.
"""
function initializer(problem::BilinearSensingProblem, δ::Float64)
  w = problem.w + (δ / sqrt(2)) * normalize(randn(length(problem.w)))
  x = problem.x + (δ / sqrt(2)) * normalize(randn(length(problem.x)))
  return [w; x]
end

function bilinear_sensing_problem(m::Int, d::Int)
  L = randn(m, d)
  R = randn(m, d)
  w = normalize(randn(d))
  x = normalize(randn(d))
  return BilinearSensingProblem(L, R, w, x, (L * w) .* (R * x))
end

"""
  generate_sparse_vector(d, k)

Generate a random length-`d` vector of unit norm with `k` nonzero elements.
"""
function generate_sparse_vector(d, k)
  x = zeros(d)
  x[sample(1:d, k, replace=false)] = normalize(randn(k))
  return x
end

struct MaxAffineRegressionProblem
  A  :: Matrix{Float64}
  # True slopes of each affine piece, one per column.
  βs :: Matrix{Float64}
  y  :: Vector{Float64}
end

function loss(problem::MaxAffineRegressionProblem)
  A = problem.A
  m = length(problem.y)
  d, k = size(problem.βs)
  # Assumes the input is a flattened version of `βs`, so it must be reshaped
  # before applying the operator `A`.
  return z -> (1 / m) * norm(maximum(A * reshape(z, d, k), dims=2)[:] .- y, 1)
end

function subgradient(problem::MaxAffineRegressionProblem)
  y = problem.y
  A = problem.A
  d, k = size(problem.βs)
  grad_fn(z) = begin
    Z = reshape(z, d, j)
    signs = sign.(maximum(A * Z, dims=2)[:] .- y)
    inds  = Int.(A * Z .== maximum(A * Z, dims=2)[:])
    # Without vectorizing, result would be a `d × 1` matrix.
    return ((1 / length(y)) * A' * (signs .* inds))[:]
  end
end

"""
  initializer(problem::MaxAffineRegressionProblem, δ::Float64)

Return an initial estimate of the slopes of each affine piece for a max-affine
regression `problem` with normalized distance `δ` from the ground truth.
"""
function initializer(problem::MaxAffineRegressionProblem, δ::Float64)
  Δ = randn(size(problem.βs))
  return problem.βs .+ δ .* (Δ ./ norm(Δ))
end

function max_affine_regression_problem(m, d, k)
  A = randn(m, d)
  βs = mapslices(normalize, randn(d, k), dims=1)
  return MaxAffineRegressionProblem(A, βs, maximum(A * βs, dims=2)[:])
end

struct LassoProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  λ::Float64
end

"""
  ℓ₁prox(x, λ)

Return the proximal operator of `|x|₁` with proximal parameter `λ`.
"""
function ℓ₁prox(x, λ)
  return sign.(x) .* max.(abs.(x) .- λ, 0)
end

"""
  estimate_params(problem::LassoProblem) -> (λ, τ)

Estimate the parameters `λ` (ℓ₁ norm penalty) and `τ` (gradient step size)
for a LASSO regression `problem`.
"""
function estimate_params(problem::LassoProblem)
  return (0.1 * norm(problem.A' * problem.y, Inf), 1 / (opnorm(problem.A)^2))
end

"""
  residual(problem::LassoProblem, τ::Float64)

Compute the residual `I - T`, where `T` is the proximal gradient operator
for a LASSO problem with ℓ₁ penalty `τ`.
"""
function residual(problem::LassoProblem, τ::Float64)
  A = problem.A
  y = problem.y
  λ = problem.λ
  return z -> z - ℓ₁prox(z - τ .* A' * (A * z .- y), λ * τ)
end

"""
  jacobian(problem::LassoProblem, τ::Float64)

Compute the Jacobian of `I - T`, where `T` is the proximal gradient operator
for a LASSO problem with ℓ₁ penalty `τ`.
"""
function jacobian(problem::LassoProblem, τ::Float64)
  A = problem.A
  y = problem.y
  λ = problem.λ
  return z -> I - Diagonal(abs.(z .- (τ .* A' * (A * z .- y))) .≥ λ * τ) * (I - τ .* A'A)
end

function lasso_problem(m, d, k, λ)
  # Use a well-conditioned design matrix.
  A = Matrix(qr(randn(d, m)).Q)'
  x = generate_sparse_vector(d, k)
  y = A * x
  return LassoProblem(A, x, y, 0.1 * norm(A'y, Inf))
end

# TODO:
# 1) sparse logistic regression
# 2) linear / quadratic programming
# 3) Set intersection problems
