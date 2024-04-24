%formhe-groundtruth:../../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=
%formhe-selfeval-lines:1 3 4
%formhe-selfeval-fix:vertex(X) :- e(X, Y). 1 { assign(V, C) : color(C) } 1 :- vertex(V).  :- e(X, Y), assign(X, C), assign(Y, C).

color((1..k)).
vertex(X) :- assign(X, Y).
vertex(Y) :- e(X, Y).
1 { assign(V, C) : color(C) } k :- vertex(V).
 :- e(X, Y), assign(X, C), Y == C.