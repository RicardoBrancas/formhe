%formhe-groundtruth:../01/A_0_1.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=

% Default
%% #const k = 3.

% (Directed) Edges
%% e(a,b). e(a,c). e(b,c). e(d,c).

% Nodes
node(A) :- e(A,_).
node(B) :- e(_,B).

% Generate
{ assign(X,1..k) } = 1 :- node(X).
% Test / Integretity Constraint
:- e(X,Y), assign(X,C), assign(Y,C).

% Display
#show assign/2.