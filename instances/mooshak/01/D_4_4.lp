%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=

% % Mooshak-exercises in lab13 from ALC
% % Problem D
% % constants
% % #const k = 2 .

% % facts
% e(1,2). 
% e(1,3). 
% e(4,3). 
% e(4,5).

% limit(k).
% generator
c(k)

sel(X) :- e(X, Y), not sel(Y) .
solve :- limit(c), #count{ vertex(X) : sel(X) } <= c .


% tester
:- not solve .