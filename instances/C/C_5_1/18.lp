%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:s(X) :- e(_, X). k { sel(X) : s(X) } k. cov(I) :- sel(X), e(I, X), i(I).

i(X) :- e(X, _).
s(X) :- e(I0, X).
k { s(X) : s(X) } 0.
cov(I) :- sel(X), e(I, X), i(I1).
 :- not cov(X), i(X).