%formhe-groundtruth:../01/B_0_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShhLCAzKS4KZShiLCA0KS4KZShiLCA1KS4KZShiLCA2KS4KZShjLCA3KS4KZShjLCA4KS4=

set(S) :- e(S,X).
:- e(A,X), e(B,X), sel(A), sel(B), A != B.
k{ sel(S) : set(S) }k.
#show sel/1.