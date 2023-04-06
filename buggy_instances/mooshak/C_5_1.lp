%formhe-groundtruth:../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=

i(X) :- e(X,_).
s(X) :- e(_, X).

k{sel(X): s(X)}k.

cov(I) :- sel(X), e(I, X), i(I).

:- not cov(X), i(X).

#show sel/1.