%formhe-groundtruth:../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-selfeval-lines:0
%formhe-selfeval-fix:k {sel(A) : e(Y,A)} k. cov(X) :- e(X, S), sel(S). :- not cov(X), e(X, _).

sel(A) :- e(Y,A), e(Y,B), not sel(B).