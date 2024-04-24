%formhe-groundtruth:./D_0_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=

% % Mooshak-exercises in lab13 from ALC
% % Problem D
% % constants
% #const k = 2 .

% % facts
% e(1,2) . 
% e(1,3) . 
% e(4,3) . 
% e(4,5) .

% generator
v(X):- e(_,X) .
v(X):- e(X,_) .

{ sel(V): v(V) } k .


% tester
%% stating: if vertex1 and vertex2 are connected 
%%  through an edge, then none of them must be 
%%  selected
:- e(V1, V2), not sel(V1), not sel(V2) .