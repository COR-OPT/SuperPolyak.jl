struct LinearProgram
  A::Matrix{Float64}
  b::Vector{Float64}
  c::Vector{Float64}
  x::Vector{Float64}
end

default_tau(problem::LinearProgram) = 0.9 / opnorm(problem.A)

"""
  seminorm(problem::LinearProgram, z::Vector{Float64}, τ::Float64 = default_tau(problem))

Compute the Chambolle-Pock seminorm using stepsize `τ` at `z`.
"""
function seminorm(problem::LinearProgram, z::Vector{Float64}, τ::Float64 = default_tau(problem))
  m, d = size(problem.A)
  x = z[1:d]; y = z[(d+1):end]
  return sqrt((1 / τ) * (norm(x)^2 + norm(y)^2) - 2 * y' * (problem.A * x))
end

"""
  chambolle_pock(problem::LinearProgram, z::Vector{Float64}, τ::Float64 = default_tau(problem))

Run a step of the Chambolle-Pock algorithm to solve the LP `problem` at
`z` with stepsize `τ`.
"""
function chambolle_pock(problem::LinearProgram, z::Vector{Float64}, τ::Float64 = default_tau(problem))
  A = problem.A
  b = problem.b
  c = problem.c
  m, d = size(A)
  x = z[1:d]
  y = z[(d+1):end]
  return [max.(0, x + τ * (A' * (y - 2τ * (A * x - b)) - c));
          y - τ * (A * x - b)]
end

"""
  loss(problem::LinearProgram, τ::Float64 = default_tau(problem))

Return a callable implementing the loss for a linear programming problem, which
is the residual measured in the Chambolle-Pock seminorm:

  |M^{1/2} (z - T(z))|,   where   M = [(1/τ)I -A'; -A (1/τ)I].
"""
function loss(problem::LinearProgram, τ::Float64 = default_tau(problem))
  return z -> seminorm(problem, z - chambolle_pock(problem, z, τ), τ)
end

"""
  subgradient(problem::LinearProgram, τ::Float64 = default_tau(problem))

Return callable implementing the subgradient of `loss(problem, τ)`.
"""
function subgradient(problem::LinearProgram, τ::Float64 = default_tau(problem))
  A = problem.A
  b = problem.b
  c = problem.c
  m, d = size(A)
  # Matrix M inducing the Chambolle-Pock seminorm.
  # Convert to sparse matrix to accelerate computation.
  M = sparse(
    [(1 / τ) * LinearAlgebra.I  -A'; -A (1 / τ) * LinearAlgebra.I]
  )
  G(z) = begin
    x = z[1:d]
    y = z[(d+1):end]
    r = z - chambolle_pock(problem, z, τ)
    q = (seminorm(problem, r, τ) ≤ 1e-15) ? zeros(length(z)) : (M * r / seminorm(problem, r, τ))
    D = Diagonal([(x + τ * (A' * (y - 2τ * (A * x - b)) - c)) .≥ 0;
                  ones(m)])
    return q - [LinearAlgebra.I - 2 * τ^2 * A'A -τ*A'; τ*A LinearAlgebra.I] * (D * q)
  end
  return G
end

function random_linear_program(m::Int, d::Int)
  # 0-1 constraint matrix
  A = rand([0, 1], m, d)
  c = randn(d)
  x = max.(0, randn(d))
  b = A * x
  return LinearProgram(A, b, c, x)
end
