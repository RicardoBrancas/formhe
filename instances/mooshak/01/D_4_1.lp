%formhe-groundtruth:../../../mooshak/instances/1819_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:0 {sel(X;Y) : e(X, Y)} k. :- not sel(X1), not sel(X2), e(X1, X2).

% % Mooshak-exercises in lab13 from ALC
% % Problem D
% % constants
% #const k = 2 .

% % facts
% e(1,2). 
% e(1,3). 
% e(4,3). 
% e(4,5).

% generator
1 { sel(X) : e(X, _) } k .
1 { sel(X) : e(_, X) } k .

% tester
:- sel(X1), sel(X2), e(X1, _), e(X2, _), X1 != X2 .