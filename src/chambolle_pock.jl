struct QuadraticProgram
  A::AbstractMatrix{Float64}
  P::AbstractMatrix{Float64}
  b::Vector{Float64}
  c::Vector{Float64}
  l::Vector{Float64}
  u::Vector{Float64}
end

struct StepSize
  τ::Float64
  σ::Float64
end

"""
  spectral_norm(A::AbstractMatrix{Float64})

Compute the spectral norm of `A`. If `A` is a sparse matrix, uses `svds` to
compute the top singular value.
"""
function spectral_norm(A::AbstractMatrix{Float64})
  if A isa SparseMatrixCSC
    svd_obj, = svds(A, nsv = 1, ritzvec = false)
    return maximum(svd_obj.S)
  else
    return opnorm(A)
  end
end

"""
  default_stepsize(problem::QuadraticProgram)

Compute a pair of default stepsizes for the Condat-Vu algorithm applied to
solve a `QuadraticProgram`.
"""
function default_stepsize(problem::QuadraticProgram)
  σ = 0.9 / spectral_norm(problem.A)
  τ = 1 / (0.5 * spectral_norm(problem.P) + (1 / σ))
  return StepSize(τ, σ)
end

"""
  seminorm(problem::QuadraticProgram, z::Vector{Float64}, step::StepSize= default_stepsize(problem))

Compute the Chambolle-Pock seminorm using stepsize `step` at `z`.
"""
function seminorm(
  problem::QuadraticProgram,
  z::Vector{Float64},
  step::StepSize = default_stepsize(problem),
)
  m, d = size(problem.A)
  τ, σ = step.τ, step.σ
  x = z[1:d]
  y = z[(d+1):end]
  return sqrt(
    (1 / τ) * (norm(x)^2) + (1 / σ) * (norm(y)^2) - 2 * y' * (problem.A * x)
  )
end

"""
  proj_box(z::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64})

Project the vector `z` to the box `[ℓ, u]`.
"""
function proj_box(z::Vector{Float64}, l::Vector{Float64}, u::Vector{Float64})
  return max.(min.(z, u), l)
end

function chambolle_pock_iteration(
  problem::QuadraticProgram,
  z::Vector{Float64},
  step::StepSize = default_stepsize(problem),
)
  m, d = size(problem.A)
  τ, σ = step.τ, step.σ
  x = z[1:d]
  y = z[(d+1):end]
  Ax = problem.A * x
  Px = problem.P * x
  return [
    proj_box(
      x - τ * (Px + problem.c + problem.A' * (y + 2σ * (Ax - problem.b))),
      problem.l,
      problem.u,
    );
    y + σ * (Ax - problem.b)
  ]
end

"""
  chambolle_pock_loss(problem::QuadraticProgram, step::StepSize) -> Function

Return a callable that computes the residual of one iteration of Chambolle-Pock
applied to `problem` with step size `step`.
"""
function chambolle_pock_loss(
  problem::QuadraticProgram,
  step::StepSize = default_stepsize(problem),
)
  A = problem.A
  P = problem.P
  b = problem.b
  c = problem.c
  l = problem.l
  u = problem.u
  τ, σ = step.τ, step.σ
  m, d = size(A)
  x = zeros(d)
  y = zeros(m)
  loss_fn(z) = begin
    x = z[1:d]
    y = z[(d+1):end]
    Ax = A * x
    x_diff = x - max.(
      l,
      min.(u, x - τ * (P * x + c + A' * (y + 2σ * (Ax - b)))),
    )
    y_diff = σ * (b - Ax)
    return sqrt(
      (1 / τ) * norm(x_diff)^2 + (1 / σ) * norm(y_diff)^2 -
      2 * y_diff' * A * x_diff
    )
  end
  return loss_fn
end

"""
  kkt_error(problem::QuadraticProgram)

Return a callable that computes the KKT error of a `QuadraticProgram`.
"""
function kkt_error(problem::QuadraticProgram)
  A = problem.A
  P = problem.P
  b = problem.b
  c = problem.c
  l = problem.l
  u = problem.u
  m, d = size(A)
  finite_l_ind = isfinite.(l)
  finite_u_ind = isfinite.(u)
  l_finite = zeros(d)
  u_finite = zeros(d)
  @. l_finite[finite_l_ind] = l[finite_l_ind]
  @. u_finite[finite_u_ind] = u[finite_u_ind]
  # Indices where λ = Px + c + A'y should be positive, negative
  # and zero, respectively.
  pos_ind = @. finite_l_ind & !finite_u_ind
  neg_ind = @. finite_u_ind & !finite_l_ind
  nul_ind = @. !(finite_l_ind | finite_u_ind)
  x = zeros(d)
  y = zeros(m)
  loss_fn(z) = begin
    x = z[1:d]
    y = z[(d+1):end]
    Ax = A * x
    Px = P * x
    # Parameter λ
    λ = Px + c + A'y
    # Duality gap
    gap = abs(x'Px + c'x + b'y + min.(λ, 0.0)' * u_finite - max.(λ, 0.0)' * l_finite)
    # Primal residual
    pri_res = norm(Ax - b) + norm(max.(x - u, 0.0)) + norm(max.(l - x, 0.0))
    # Dual residual |λ - P_{Λ}(λ)|
    dua_res = norm(max.(λ .* pos_ind, 0.0)) + norm(min.(λ .* neg_ind, 0.0)) + norm(λ .* nul_ind)
    return gap + pri_res + dua_res
  end
  return loss_fn
end

"""
  kkt_error_subgradient(problem::QuadraticProgram)

Return a callable that computes a subgradient of the KKT error of a `QuadraticProgram`.
"""
function kkt_error_subgradient(problem::QuadraticProgram)
  m, d = size(problem.A)
  compiled_loss_tape = compile(GradientTape(kkt_error(problem), randn(m + d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  chambolle_pock_subgradient(problem::QuadraticProgram, step::StepSize) -> Function

Return a callable that computes a subgradient of the Chambolle-Pock residual
for `problem` with step size `step`.
"""
function chambolle_pock_subgradient(
  problem::QuadraticProgram,
  step::StepSize = default_stepsize(problem),
)
  m, d = size(problem.A)
  compiled_loss_tape = compile(GradientTape(chambolle_pock_loss(problem, step), randn(m + d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  read_problem_from_qps_file(filename::String) -> QuadraticProgram

Read a quadratic programming problem from a .QPS file and convert it to a
`QuadraticProgram` struct. Assumes that the input problem has equality
constraints of the form Ax = b.
"""
function read_problem_from_qps_file(filename::String, mpsformat=:fixed)
  problem = readqps(filename, mpsformat=mpsformat)
  m, n = problem.ncon, problem.nvar
  # The objective matrix is symmetric and the .QPS file gives
  # the lower-triangular part only.
  P = sparse(problem.qrows, problem.qcols, problem.qvals, n, n)
  P = P + tril(P, 1)'
  A = sparse(problem.arows, problem.acols, problem.avals, m, n)
  b = problem.ucon
  @assert norm(problem.lcon - problem.ucon, Inf) ≤ 1e-15 "Cannot convert to equality form"
  return QuadraticProgram(
    A,
    P,
    problem.ucon,
    problem.c,
    problem.lvar,
    problem.uvar,
  )
end
