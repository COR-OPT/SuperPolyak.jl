function soft_threshold(x::Vector{Float64}, τ::Float64)
  return sign.(x) .* max.(abs.(x) .- τ, 0.0)
end

struct LassoProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  λ::Float64
end

"""
  proximal_gradient(problem::LassoProblem, x::Vector{Float64}, τ::Float64)

Compute the proximal operator for the Lasso problem

  min_x |Ax - b|^2 + λ |x|₁,

with step size `τ`.
"""
function proximal_gradient(
  problem::LassoProblem,
  x::Vector{Float64},
  τ::Float64,
)
  return soft_threshold(
    x - τ * problem.A' * (problem.A * x - problem.y),
    problem.λ * τ,
  )
end

"""
  loss(problem::LassoProblem, τ::Float64 = 0.95 / (opnorm(problem.A)^2))

Compute the residual of the proximal gradient method applied to solve a LASSO
regression `problem` with step `τ`.
"""
function loss(problem::LassoProblem, τ::Float64 = 0.95 / (opnorm(problem.A)^2))
  A = problem.A
  y = problem.y
  λ = problem.λ
  loss_fn(z) = begin
    grad_step = z - τ * A' * (A * z - y)
    return norm(z - sign.(grad_step) .* max.(abs.(grad_step) .- λ * τ, 0.0))
  end
  return loss_fn
end

function subgradient(
  problem::LassoProblem,
  τ::Float64 = 0.95 / (opnorm(problem.A)^2),
)
  d = size(problem.A, 2)
  # Define function here.
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  support_recovery_lambda(A::Matrix{Float64}, x::Vector{Float64},
                          σ::Float64)

Compute a value for `λ` that guarantees recovery of a support contained within
the support of the ground truth solution under normalized Gaussian designs.
"""
function support_recovery_lambda(
  A::Matrix{Float64},
  x::Vector{Float64},
  σ::Float64,
)
  m, d = size(A)
  nnz_ind = abs.(x) .> 1e-15
  # Compute factor γ = 1 - |X_{S^c}'X_S (X_S'X_S)^{-1}|_{∞}.
  S = A[:, nnz_ind]
  T = A[:, .!nnz_ind]
  γ = 1.0 - opnorm(T' * (S * inv(S'S)), Inf)
  @info "γ = $(γ)"
  return (2.0 / γ) * sqrt(σ^2 * log(d) / m)
end

function lasso_problem(m::Int, d::Int, k::Int, σ::Float64 = 0.1; kwargs...)
  x = generate_sparse_vector(d, k)
  A = Matrix(qr(randn(d, m)).Q)'
  y = A * x + σ .* randn(m)
  return LassoProblem(A, x, y, get(kwargs, :λ, 0.2 * norm(A'y, Inf)))
end

function compute_tau(problem::LassoProblem)
  return 0.95 / (opnorm(problem.A)^2)
end

function initializer(problem::LassoProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end
