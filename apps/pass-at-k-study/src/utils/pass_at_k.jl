# Unbiased pass@k estimator (sampling without replacement).
# pass@k = 1 - C(n-c, k) / C(n, k) where n = total samples, c = correct samples.

function binomial_coeff(n::Int, k::Int)
    k < 0 && return 0
    k > n && return 0
    k == 0 && return 1
    # C(n,k) = n! / (k! * (n-k)!)
    k = min(k, n - k)
    numer = one(n)
    for i in 0:(k - 1)
        numer *= (n - i)
    end
    denom = factorial(k)
    return numer ÷ denom
end

"""
    pass_at_k_unbiased(n_total, n_correct, k)

Unbiased estimate of pass@k: probability that at least one of k draws (without replacement)
from n_total samples is correct, given n_correct correct samples.

Returns a Float64 in [0, 1]. Returns NaN if n_total < k or n_correct > n_total.
"""
function pass_at_k_unbiased(n_total::Int, n_correct::Int, k::Int)
    k <= 0 && return 1.0
    n_total < k && return NaN
    n_correct > n_total && return NaN
    n_incorrect = n_total - n_correct
    if n_incorrect < k
        return 1.0  # more correct than k, so pass@k = 1
    end
    # pass@k = 1 - C(n_incorrect, k) / C(n_total, k)
    c_incorrect = binomial_coeff(n_incorrect, k)
    c_total = binomial_coeff(n_total, k)
    c_total == 0 && return NaN
    1.0 - (c_incorrect / c_total)
end
