%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E).

element(E) :- e(E, X).
set(X) :- e(E, X).
 :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), Y != k, element(E).
k { sel(X) : set(X) } k.