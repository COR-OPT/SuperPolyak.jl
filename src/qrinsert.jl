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
function qrinsert!(
  Q::Matrix{Float64},
  R::Matrix{Float64},
  v::Vector{Float64},
)
  m = size(Q, 1)
  n = size(R, 1)
  # Assumption: `v` is added at the end of our matrix.
  w = Q' * v
  for idx in (m-1):-1:(n+1)
    # Calculate and apply Givens rotation to extra column.
    G = givens(w, idx, idx+1)
    lmul!(G[1], w)
    # Apply Givens rotation to Q from the right
    rapply_givens!(Q, G[1])
  end
  return [[R; zeros(1, n)] w[1:(n+1)]]
end
