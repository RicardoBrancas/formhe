%formhe-groundtruth:../../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix:node(I0) :- e(_, I0). node(I0) :- e(I0, _). :- e(X,Y), assign(X,C), assign(Y,C).

% Mooshak-exercises in lab13 from ALC
% Problem A

% % Default
% #const n = 3.
% colours(1..k)

% Generate
% { assign(X,1..coulours) } = 1 :- node(X).
% Test
% :- e(X,Y), assign(X,C), assign(Y,C).

% % Nodes
% node(1..6).
% Edges
% e(1,(2;3;4)).  e(2,(4;5;6)).  e(3,(1;4;5)).
% e(4,(1;2)).    e(5,(3;4;6)).  e(6,(2;3;5)).

% % Display
% #show color/2.

% facts
col(1..k).

% generate
{ assign(X,C) : col(C) } == 1 :- node(X).

% constraint
:- edge(X,Y), assign(X,C), assign(Y,C).