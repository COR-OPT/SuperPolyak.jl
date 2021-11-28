function soft_threshold(x::Vector{Float64}, τ::Float64)
  return sign.(x) .* max.(abs.(x) .- τ, 0.0)
end

struct LassoProblem
  A :: Matrix{Float64}
  x :: Vector{Float64}
  y :: Vector{Float64}
  λ :: Float64
end

"""
  proximal_gradient(A::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64}, λ::Float64, τ::Float64)

Compute the proximal gradient operator for the LASSO problem

  min_x |Ax - b|^2 + λ * |x|_1

with step `τ`.
"""
function proximal_gradient(
  A::Matrix{Float64},
  x::Vector{Float64},
  y::Vector{Float64},
  λ::Float64,
  τ::Float64,
)
  return soft_threshold(x - τ * A' * (A * x - y), λ * τ)
end

function loss(problem::LassoProblem, τ::Float64)
  return z -> norm(z - proximal_gradient(problem.A, z, problem.y, problem.λ, τ))
end

function subgradient(problem::LassoProblem, τ::Float64)
  A = problem.A
  y = problem.y
  λ = problem.λ
  g(z) = begin
    r = z - proximal_gradient(A, z, y, λ, τ)
    # Nonzero indices in diagonal
    D = Diagonal(abs.(z - τ * A' * (A * z - y)) .≥ λ * τ)
    J = LinearAlgebra.I - (D - τ * A' * (A * D))
    return (norm(r) ≤ 1e-15) ? zeros(length(z)) : J' * normalize(r)
  end
  return g
end

function lasso_problem(m::Int, d::Int, k::Int, σ::Float64 = 0.1; kwargs...)
  x = generate_sparse_vector(d, k)
  A = Matrix(qr(randn(d, m)).Q)'
  y = A * x + σ .* randn(m)
  return LassoProblem(A, x, y, get(kwargs, :λ, norm(A'y, Inf) / 10))
end

function compute_tau(problem::LassoProblem)
  return 0.5 / (opnorm(problem.A)^2)
end

function initializer(problem::LassoProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end
