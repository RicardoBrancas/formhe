%formhe-groundtruth:../../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { assign(A, (1..k)) : #true } 1 :- e(B, A).

 :- e(A, B), assign(A, C), assign(B, C).
1 { assign(A, (1..k)) : #true } 1 :- e(A, B).
1 { assign(A, (1..k)) : #true } 1 :- assign(C, A).