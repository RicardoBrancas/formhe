%formhe-groundtruth:../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

s(X) :- e(X,E).
k { sel(X) : s(X) } k.
inter(X, Y) :- e(X, E), e(Y, E), X != Y. :- inter(X, Y), sel(X), sel(Y).