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

function loss(problem::LassoProblem, τ::Float64 = 0.9 / (opnorm(problem.A)^2))
  return z -> norm(z - proximal_gradient(problem.A, z, problem.y, problem.λ, τ))
end

function subgradient(problem::LassoProblem, τ::Float64 = 0.9 / (opnorm(problem.A)^2))
  A = problem.A
  y = problem.y
  λ = problem.λ
  g(z) = begin
    r = z - proximal_gradient(A, z, y, λ, τ)
    q = (norm(r) ≤ 1e-14) ? zeros(length(z)) : normalize(r)
    # Nonzero indices in diagonal
    D = Diagonal(abs.(z - τ * A' * (A * z - y)) .≥ λ * τ)
    return q - (D * q - τ * A'A * (D * q))
  end
  return g
end

"""
  support_recovery_lambda(A::Matrix{Float64}, x::Vector{Float64},
                          σ::Float64)

Compute a value for `λ` that guarantees recovery of a support contained within
the support of the ground truth solution under normalized Gaussian
designs.
"""
function support_recovery_lambda(
  A::Matrix{Float64},
  x::Vector{Float64},
  σ::Float64,
)
  m, d = size(A)
  nnz_ind = abs.(x) .> 1e-15
  # Compute factor γ = 1 - |X_{S^c}'X_S (X_S'X_S)^{-1}|_{∞}.
  S = view(A, :, nnz_ind)
  T = view(A, :, .!nnz_ind)
  γ = 1.0 - opnorm(T' * (S * inv(S'S)), Inf)
  @info "γ = $(γ)"
  return (2.0 / γ) * sqrt(σ^2 * log(d) / m)
end

function lasso_problem(m::Int, d::Int, k::Int, σ::Float64 = 0.01; kwargs...)
  x = generate_sparse_vector(d, k)
  A = Matrix(qr(randn(d, m)).Q)'
  # A ./= sqrt.(sum(A.^2, dims=1))
  y = A * x + σ .* randn(m)
  return LassoProblem(A, x, y, get(kwargs, :λ, 0.2 * norm(A'y, Inf)))
end

function compute_tau(problem::LassoProblem)
  return 0.9 / (opnorm(problem.A)^2)
end

function initializer(problem::LassoProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end
