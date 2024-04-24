%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:element(E) :- e(E, X).  :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E).

element(E) :- e(E, _).
set(X) :- e(E, X).
 :- sel(X), sel(Y), not e(E, X), -e(E, Y), X == Y, set(X), set(Y), element(E), I0 == 1.
k { sel(X) : set(X) } k.