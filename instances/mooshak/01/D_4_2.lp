%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=
%formhe-selfeval-lines:0
%formhe-selfeval-fix:{ sel(X;Y) : e(X,Y) } k. :- not sel(X), not sel(Y), e(X,Y).

% Mooshak-exercises in lab13 from ALC
% Problem D


limit(2).
% generator
sel(X) :- e(X, Y), not sel(Y) .
solve :- limit(C), #count{ vertex(X) : sel(X) } <= C .


% tester
:- not solve .