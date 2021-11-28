"""
  get_cumul_oracle_calls(oracle_calls::Vector{Int}, show_amortized::Bool)

Return a vector with the cumulative number of oracle calls given a history of
oracle calls per iteration of the algorithm. If `show_amortized` is `true`, the
total number is divided equally among each step.
"""
function get_cumul_oracle_calls(oracle_calls::Vector{Int}, show_amortized::Bool)
  T = length(oracle_calls)
  cumul_oracle_calls =
    show_amortized ? ((0:(T-1)) .* (sum(oracle_calls) ÷ (T-1))) : cumsum(oracle_calls)
  return cumul_oracle_calls
end
