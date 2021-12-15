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
  return z ->
    (1 / length(problem.y)) * norm((problem.A * z) .^ 2 .- problem.y, 1)
end

function subgradient(problem::PhaseRetrievalProblem)
  m, d = size(problem.A)
  compiled_loss_tape = compile(
    GradientTape(loss(problem), rand(d)),
  )
  return z -> gradient!(compiled_loss_tape, z)
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
  return PhaseRetrievalProblem(A, x, (A * x) .^ 2)
end

"""
  QuadraticSensingProblem

A symmetrized quadratic sensing problem with measurements

  yᵢ = |pᵢ'X|² - |qᵢ'X|² = (pᵢ - qᵢ)'XX'(pᵢ + qᵢ)

where pᵢ and qᵢ are iid Gaussian vectors.

We let ℓᵢ = pᵢ - qᵢ, rᵢ = pᵢ + qᵢ below.
"""
struct QuadraticSensingProblem
  L::Matrix{Float64}
  R::Matrix{Float64}
  X::Matrix{Float64}
  y::Vector{Float64}
end

function loss(problem::QuadraticSensingProblem)
  d, k = size(problem.X)
  m = length(problem.y)
  L = problem.L
  R = problem.R
  y = problem.y
  loss_fn(z) = begin
    Z = reshape(z, d, k)
    return (1 / m) * norm(y - sum((L * Z) .* (R * Z), dims = 2)[:], 1)
  end
  return loss_fn
end

function subgradient(problem::QuadraticSensingProblem)
  d, r = size(problem.X)
  compiled_loss_tape = compile(
    GradientTape(loss(problem), rand(d * r)),
  )
  return z -> gradient!(compiled_loss_tape, z)
end

function initializer(problem::QuadraticSensingProblem, δ::Float64)
  Δ = randn(size(problem.X))
  return problem.X + δ * (Δ / norm(Δ))
end

function quadratic_sensing_problem(m::Int, d::Int, r::Int)
  X = Matrix(qr(randn(d, r)).Q)
  L = sqrt(2) * randn(m, d)
  R = sqrt(2) * randn(m, d)
  y = sum((L * X) .* (R * X), dims = 2)[:]
  return QuadraticSensingProblem(L, R, X, y)
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
  d = length(problem.w)
  compiled_loss_tape = compile(GradientTape(loss(problem), rand(d)))
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  initializer(problem::BilinearSensingProblem, δ::Float64)

Return an initial estimate `(w₀, x₀)` that is `δ`-close to the ground truth
in normalized distance. Output a single vector containing `w₀` and `x₀` stacked
vertically.
"""
function initializer(problem::BilinearSensingProblem, δ::Float64)
  wx_stacked = [problem.w; problem.x]
  return wx_stacked + δ * normalize(randn(length(wx_stacked)))
end

"""
  bilinear_sensing_problem(m::Int, d::Int)

Generate a bilinear sensing problem in `d` dimensions with `m` measurements
using random Gaussian sensing matrices.
"""
function bilinear_sensing_problem(m::Int, d::Int)
  L = randn(m, d)
  R = randn(m, d)
  w = normalize(randn(d))
  x = normalize(randn(d))
  return BilinearSensingProblem(L, R, w, x, (L * w) .* (R * x))
end

"""
  GeneralBilinearSensingProblem

A bilinear sensing problem with measurements

  `yᵢ = ℓᵢ'W * X'rᵢ`

where `W` and `X` are `d × r` matrices.
"""
struct GeneralBilinearSensingProblem
  L::Matrix{Float64}
  R::Matrix{Float64}
  W::Matrix{Float64}
  X::Matrix{Float64}
  y::Vector{Float64}
end

"""
  loss(problem::GeneralBilinearSensingProblem)

Implement the ℓ₁ robust loss for a bilinear sensing problem with general rank.
Assumes that the argument will be a vector containing the "flattened" version
of the matrix `[W, X]`, where `W` and `X` are `d × r` matrices.
"""
function loss(problem::GeneralBilinearSensingProblem)
  L = problem.L
  R = problem.R
  y = problem.y
  m = length(y)
  d, r = size(problem.W)
  return z -> begin
    # Layout assumption: z is a "flattened" version of [W, X] ∈ Rᵈˣ⁽²ʳ⁾.
    W = reshape(z[1:(d*r)], d, r)
    X = reshape(z[(d*r+1):end], d, r)
    # Compute row-wise product.
    return (1 / m) * norm(y .- sum((L * W) .* (R * X), dims = 2)[:], 1)
  end
end

"""
  subgradient(problem::GeneralBilinearSensingProblem)

Implement the subgradient of the ℓ₁ robust loss for a bilinear sensing problem
with general rank. Like `loss(problem)`, assumes that the argument will be a
vector containing the "flattened" version of the matrix `[W, X]`, where `W` and
`X` are `d × r` matrices.
"""
function subgradient(problem::GeneralBilinearSensingProblem)
  d, r = size(problem.W)
  compiled_loss_tape = compile(
    GradientTape(loss(problem), rand(2 * d * r)),
  )
  return z -> gradient!(compiled_loss_tape, z)
end

"""
  general_bilinear_sensing_problem(m::Int, d::Int, r::Int)

Generate a bilinear sensing problem with solutions of dimension `d × r` and `m`
measurements using random Gaussian sensing matrices.
"""
function general_bilinear_sensing_problem(m::Int, d::Int, r::Int)
  L = randn(m, d)
  R = randn(m, d)
  # Solutions on the orthogonal manifold O(d, r).
  W = Matrix(qr(randn(d, r)).Q)
  X = Matrix(qr(randn(d, r)).Q)
  y = sum((L * W) .* (R * X), dims = 2)[:]
  return GeneralBilinearSensingProblem(L, R, W, X, y)
end

"""
  initializer(problem::GeneralBilinearSensingProblem, δ::Float64)

Generate an initial guess for the solution to `problem` that is `δ`-far from
the ground truth when distance is measured in the Euclidean norm.
"""
function initializer(problem::GeneralBilinearSensingProblem, δ::Float64)
  wx_stacked = [vec(problem.W); vec(problem.X)]
  return wx_stacked + δ * normalize(randn(length(wx_stacked)))
end

"""
  generate_sparse_vector(d, k)

Generate a random length-`d` vector of unit norm with `k` nonzero elements.
"""
function generate_sparse_vector(d, k)
  x = zeros(d)
  x[sample(1:d, k, replace = false)] = normalize(randn(k))
  return x
end

struct MaxAffineRegressionProblem
  A::Matrix{Float64}
  # True slopes of each affine piece, one per column.
  βs::Matrix{Float64}
  y::Vector{Float64}
end

function loss(problem::MaxAffineRegressionProblem)
  A = problem.A
  y = problem.y
  m = length(y)
  d, k = size(problem.βs)
  # Assumes the input is a flattened version of `βs`, so it must be reshaped
  # before applying the operator `A`.
  return z -> (1 / m) * norm(maximum(A * reshape(z, d, k), dims = 2)[:] - y, 1)
end

function subgradient(problem::MaxAffineRegressionProblem)
  d, k = size(problem.βs)
  compiled_loss_tape = compile(
    GradientTape(loss(problem), rand(d * k)),
  )
  return z -> gradient!(compiled_loss_tape, z)
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
  βs = mapslices(normalize, randn(d, k), dims = 1)
  return MaxAffineRegressionProblem(A, βs, maximum(A * βs, dims = 2)[:])
end

struct CompressedSensingProblem
  A::Matrix{Float64}
  x::Vector{Float64}
  y::Vector{Float64}
  k::Int
end

function proj_sparse(x::Vector{Float64}, k::Int)
  x[sortperm(abs.(x), rev = true)[(k+1):end]] .= 0
  return x
end

function dist_sparse(x::Vector{Float64}, k::Int)
  return norm(x - proj_sparse(x[:], k))
end

function grad_sparse(x::Vector{Float64}, k::Int)
  x₊ = proj_sparse(x[:], k)
  ds = norm(x₊ - x)
  return (ds ≤ 1e-15) ? zeros(length(x)) : (x - x₊) / ds
end

function proj_range(A::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64})
  return x + A \ (y - A * x)
end

function dist_range(A::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64})
  return norm(x - proj_range(A, x, y))
end

function grad_range(A::Matrix{Float64}, x::Vector{Float64}, y::Vector{Float64})
  x₊ = proj_range(A, x, y)
  ds = norm(x₊ - x)
  return (ds ≤ 1e-15) ? zeros(length(x)) : (x - x₊) / ds
end

function loss(problem::CompressedSensingProblem)
  A = problem.A
  y = problem.y
  k = problem.k
  m, d = size(A)
  return z -> dist_sparse(z, k) + dist_range(A, z, y)
end

function subgradient(problem::CompressedSensingProblem)
  A = problem.A
  y = problem.y
  k = problem.k
  m, d = size(A)
  return z -> grad_sparse(z, k) + grad_range(A, z, y)
end

function compressed_sensing_problem(m, d, k)
  A = randn(m, d)
  x = generate_sparse_vector(d, k)
  return CompressedSensingProblem(A, x, A * x, k)
end

"""
  initializer(problem::CompressedSensingProblem, δ::Float64)

Return an initial estimate of the solution of a compressed sensing `problem`
with normalized distance `δ` from the ground truth.
"""
function initializer(problem::CompressedSensingProblem, δ::Float64)
  return problem.x + δ * normalize(randn(length(problem.x)))
end
