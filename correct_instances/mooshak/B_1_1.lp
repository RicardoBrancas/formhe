%formhe-groundtruth:../../mooshak/instances/1920_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

-sel(A) :- e(A,Y), e(B,Y), sel(B), A!=B.
k {sel(A) : e(A,Y)} k.