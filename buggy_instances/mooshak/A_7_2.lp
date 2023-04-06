%formhe-groundtruth:../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=

% K-coloring problem
color(1..k).


vertex(X) :- e(X,_).
vertex(X) :- e(_,X).

% Generation

%% Assign one color to each vertex

1{assign(X,C) : color(C)}1 :- vertex(X).

% Test

:- e(X, Y), assign(X,C), assign(Y,C) , color(C).
:- e(Y, X), assign(X,C), assign(Y,C) , color(C).