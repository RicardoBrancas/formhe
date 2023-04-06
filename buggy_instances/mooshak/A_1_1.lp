%formhe-groundtruth:../../mooshak/instances/1920_2.lp
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=
%formhe-selfeval-lines:
%formhe-selfeval-fix:{ assign(A,1..k) } = 1 :- e(_,A).

:- e(A,B), assign(A, C), assign(B, C).
{ assign(A,1..k) } = 1 :- e(A,_).