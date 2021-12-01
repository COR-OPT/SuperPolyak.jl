struct LinearProgram
  A::Matrix{Float64}
  b::Vector{Float64}
  c::Vector{Float64}
  x::Vector{Float64}
end

"""
  seminorm(problem::LinearProgram, τ::Float64, z::Vector{Float64})

Compute the Chambolle-Pock seminorm using stepsize `τ` at `z`.
"""
function seminorm(problem::LinearProgram, τ::Float64, z::Vector{Float64})
  m, d = size(problem.A)
  x = z[1:d]; y = z[(d+1):end]
  return (1 / τ) * (norm(x)^2 + norm(y)^2) - 2 * y' * (problem.A * x)
end

function chambolle_pock(problem::LinearProgram, τ::Float64, z::Vector{Float64})
  A = problem.A
  b = problem.b
  c = problem.c
  m, d = size(A)
  x = z[1:d]
  y = z[(d+1):end]
  return [max.(0, x + τ * (A' * (y - 2τ * (A * x - b)) - c));
          y - τ * (A * x - b)]
end

function loss(problem::LinearProgram, τ::Float64 = 0.9 / opnorm(problem.A))
  return z -> norm(z - chambolle_pock(problem, τ, z))
end

function subgradient(problem::LinearProgram, τ::Float64 = 0.9 / opnorm(problem.A))
  A = problem.A
  b = problem.b
  c = problem.c
  m, d = size(A)
  G(z) = begin
    x = z[1:d]
    y = z[(d+1):end]
    r = z - chambolle_pock(problem, τ, z)
    q = (norm(r) ≤ 1e-15) ? zeros(length(z)) : normalize(r)
    D = Diagonal([(x + τ * (A' * (y - 2τ * (A * x - b)) - c)) .≥ 0;
                  ones(m)])
    return q - D * ([LinearAlgebra.I - 2 * τ^2 * A'A τ*A'; -τ*A LinearAlgebra.I] * q)
  end
  return G
end

function random_linear_program(m::Int, d::Int)
  A = randn(m, d)
  c = randn(d)
  x = max.(0, randn(d))
  b = A * x
  return LinearProgram(A, b, c, x)
end
