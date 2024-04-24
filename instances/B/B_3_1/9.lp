%formhe-groundtruth:../../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:disjunct(X, Y) :- e(X, Z), e(X, C), e(Y, V), e(Y, W), Z != V, C != W, Z != W, Z != C, V != W, C != V, X != Y. total(S) :- S = #count { X: e(X,Y), sel(X) }.

disjunct(X, Y) :- e(X, Z), e(X, C), e(Y, V), e(V, W), Z != V, C != W, Z != W, Z != C, V != W, C != V, X != Y.
 { sel(X) : #true }  :- disjunct(X, Y).
total(S) :- sel(Count), total(S).
 :- total(X), X != k.