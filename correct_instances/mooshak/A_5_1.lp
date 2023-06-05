%formhe-groundtruth:../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=

color(1..k).



v(X) :- e(X,_).
v(X) :- e(_,X).

1{assign(V,C):color(C)}1:-v(V).

:- v(A), v(B), e(A,B), assign(A,C), assign(B,C).


#show assign/2.