%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: { sel(X) : #true }  :- e(Y, X). total(S) :- S = #count { X: e(Y,X), sel(X) }.

cover(X) :- e(X, Y), sel(Y).
 { sel(Y) : #true }  :- e(Y, X).
total(S) :- sel(Count), total(S).
 :- total(S), S != k.
 :- e(X, Y), not cover(X).