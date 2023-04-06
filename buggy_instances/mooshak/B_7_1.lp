%formhe-groundtruth:../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

set(X) :- e(X,_).

k{sel(X) : set(X)}k.

:- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.