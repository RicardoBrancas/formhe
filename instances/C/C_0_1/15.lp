%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:2 4
%formhe-selfeval-fix:0 { sel(X) : s(X) } k.  :- not cov(X), v(X).

v(X) :- e(X, _).
s(X) :- e(_, X).
0 { sel(X) : s(X) } k :- e(1, I1).
cov(X) :- v(X), e(X, S), sel(S).
 :- not sel(X), v(X), e(_, 0).