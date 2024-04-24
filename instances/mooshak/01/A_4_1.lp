%formhe-groundtruth:../../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=
%formhe-selfeval-lines:
%formhe-selfeval-fix:node(I0) :- e(_, I0). node(I0) :- e(I0, _).

% Mooshak-exercises in lab13 from ALC
% Problem A

% % Default
% #const n = 3.

% Generate
{ assign(X,1..k) } = 1 :- node(X).
% Test
:- e(X,Y), assign(X,C), assign(Y,C).

% % Nodes
% node(1..6).
% Edges


% % Display
% #show color/2.