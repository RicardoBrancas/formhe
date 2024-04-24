%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { sel(X) : vertex(X)} k.

vertex(X) :- e(X, Y).
vertex(Y) :- e(X, Y).

k { sel(X) : vertex(X)} k.

covered(X, Y) :- sel(X), e(X, Y).
covered(X, Y) :- sel(Y), e(X, Y).

:- not covered(X, Y), e(X, Y).

#show sel/1.