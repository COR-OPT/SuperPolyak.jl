"""
  householder(v::AbstractVector{Float64})

A Householder transformation to eliminate all but the first elements
of the vector `v`. Returns:

- w: The vector defining the transform `H = I - 2 w * w'`.
- β: The norm |v|_2, which survives at v[1] after applying `H`.
"""
function householder(v::AbstractVector{Float64})
  β = norm(v)
  w = copy(v)
  w[1] = -norm(v[2:end])^2 / (v[1] + β)
  return normalize(w), β
end


"""
  rapply_givens!(A::Matrix{Float64}, G::LinearAlgebra.Givens)

Compute the product A * G' using a vectorized operation, assuming the indices
of the modified columns are consecutive. Modifies the matrix `A` in place.
"""
function rapply_givens!(A::Matrix{Float64}, G::LinearAlgebra.Givens)
  c = G.c
  s = G.s
  i = G.i1
  j = G.i2
  A[:, i:j] = [c -s] .* A[:, i] .+ [s c] .* A[:, j]
end

"""
  qrinsert(Q::Matrix{Float64}, R::Matrix{Float64}, v::Vector{Float64}) -> (Qnew, Rnew)

Update the reduced QR factorization of an `m × n` matrix `A` (with `m ≥ n`)
after a column `v` is added. The runtime of the algorithm is O(mn) [1].

Returns the updated `R` (reduced) factor. The `Q` factor is modified in place.

Reference:
[1] Golub, Gene H., and Charles F. Van Loan. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013, Section 6.5.2, pp. 335–338.
"""
function qrinsert!(Q::Matrix{Float64}, R::Matrix{Float64}, v::Vector{Float64})
  m = size(Q, 1)
  n = size(R, 1)
  # Assumption: `v` is added at the end of our matrix.
  w = Q' * v
  for idx in (m-1):-1:(n+1)
    # Calculate and apply Givens rotation to extra column.
    G = givens(w, idx, idx + 1)
    lmul!(G[1], w)
    # Apply Givens rotation to Q from the right
    rapply_givens!(Q, G[1])
  end
  return [[R; zeros(1, n)] w[1:(n+1)]]
end

"""
  CompactWV

A compact WV representation for a Q matrix:

  Q = I - WV^T

where `W` and `V` are `d × r` matrices.
"""
mutable struct CompactWV
  W::Matrix{Float64}
  V::Matrix{Float64}
end

# Overloads for CompactWV type.
Base.:*(Q::CompactWV, v::Vector{Float64}) = v - Q.W * (Q.V'v)
Base.:*(Q::CompactWV, M::Matrix{Float64}) = M - Q.W * (Q.V'M)
Base.adjoint(Q::CompactWV) = CompactWV(Q.V, Q.W)

"""
  update_compact_wv!(Q::CompactWV, v::Vector{Float64}, λ::Float64)

Update the WV representation of `Q` when multiplying it with a Householder
matrix `H = I - λ * v * v'` from the right.
"""
function update_compact_wv!(Q::CompactWV, v::Vector{Float64}, λ::Float64)
  Q.W = [Q.W λ * (Q * v)]
  Q.V = [Q.V v]
end

"""
  qrinsert_wv!(Q::CompactWV, R::Matrix{Float64}, v::Vector{Float64})

Update the QR decomposition after appending a column `v`, given the compact
WV representation of the matrix `Q` and the `R` matrix.
"""
function qrinsert_wv!(Q::CompactWV, R::Matrix{Float64}, v::Vector{Float64})
  d = length(v)
  m = size(Q.W, 1)
  n = size(R, 2)
  # Assumption: `v` is added at the end of the matrix.
  w = Q'v
  # Find the Householder transform to zero out w[(n+2):end].
  q, β = householder(view(w, (n+1):d))
  # Update the CompactWV representation of Q.
  update_compact_wv!(Q, [zeros(n); q], 2.0)
  # Return the new R matrix.
  return [R [w[1:n]; β; zeros(d - n - 1)]]
end


"""
  wv_from_vector(v::Vector{Float64})

Initialize a compact WV representation given a single vector. Return the
compact WV representation as well as the initial matrix `R`.
"""
function wv_from_vector(v::Vector{Float64})
  d = length(v)
  q, β = householder(v)
  R = zeros(d, 1); R[1, 1] = β
  return CompactWV(2 * q[:, :], q[:, :]), R
end
