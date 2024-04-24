%formhe-groundtruth:./A_0_1.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLDIpLiBlKDIsMykuIGUoMyw0KS4gZSg0LDUpLiBlKDUsMSkuCmUoMSw2KS4gZSgyLDcpLiBlKDMsOCkuIGUoNCw5KSwgZSg1LDEwKS4KZSg2LDgpLiBlKDYsOSkuCmUoNyw5KS4gZSg3LDEwKS4KZSg4LDEwKS4=

% % Mooshak-exercises in lab13 from ALC
% % Problem A
#const n = 3.


k(1..n).

1 { assign(N, C) : k(C) } 1 :- e(_, N).

1 { assign(N, C) : k(C) } 1 :- e(N, _).


:- assign(N, C), assign(A, C), e(N,A).