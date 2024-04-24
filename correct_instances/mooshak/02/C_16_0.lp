%formhe-groundtruth:../01/C_0_1.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=

covered(E) :- e(E, S), sel(S).

k { sel(S) : e(E, S) } k.

:- not covered(E), e(E, S).

#show sel/1.