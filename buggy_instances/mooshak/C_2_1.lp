%formhe-groundtruth:../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=

element(E) :- e(E, X).
set(X) :- e(E, X).

:- sel(X), sel(Y), not e(E, X), not e(E, Y), X!= Y, set(X), set(Y), element(E).


k{sel(X): set(X)}k.

#show sel/1.