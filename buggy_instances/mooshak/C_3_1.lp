%formhe-groundtruth:../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=

cover(X) :- e(X,Y), sel(Y).
{sel(X)} :- e(Y,X).
total(S) :- S=#count{X : e(Y,X), sel(X)}.
:- total(S), S!=k.
:- e(X,Y), not cover(X).