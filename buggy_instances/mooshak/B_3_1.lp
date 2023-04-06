%formhe-groundtruth:../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

disjunct(X,Y) :- e(X,Z), e(X,C), e(Y,V), e(Y,W), Z!=V, C!=W, Z!=W, Z!=C, V!=W, C!=V, X!=Y.
{sel(X)} :- disjunct(X,Y).
total(S) :- S = #count{ X : e(X,Y), sel(X)}.
:- total(X), X != k.