"""
  qrinsert(Q::Matrix{Float64}, R::Matrix{Float64}, v::Vector{Float64}) -> (Qnew, Rnew)

Update the reduced QR factorization of an `m × n` matrix `A` (with `m ≥ n`)
after a column `v` is added. The runtime of the algorithm is O(mn) [1].

Returns the updated `Q` (full) and `R` (reduced) factors.

Reference:
[1] Golub, Gene H., and Charles F. Van Loan. Matrix Computations. 4th ed. Johns Hopkins University Press, 2013, Section 6.5.2, pp. 335–338.
"""
function qrinsert(
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
    (G, r) = givens(w, idx, idx+1)
    w = G * w
    # Apply Givens rotation to Q from the right
    Q = Q * G'
  end
  return Q, [[R; zeros(n)'] w[1:(n+1)]]
end
