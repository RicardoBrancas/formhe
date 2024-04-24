%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix: { sel(V) : v(V) } k.

v(X) :- e(_, X).
v(X) :- e(X, _).
1 { sel(V) : sel(V) } k.
conected(V1, V2) :- e(V1, V2).
conected(V1, V2) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), not sel(V2).