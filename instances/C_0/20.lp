%formhe-groundtruth:../../correct_instances/mooshak/C_0_1.lp
%formhe-selfeval-lines:2
%formhe-selfeval-deleted-lines:0
%formhe-selfeval-changes-generate:2 2
%formhe-selfeval-changes-generate-n:2
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:
%formhe-selfeval-changes-test-n:0
%formhe-selfeval-changes-test-n-unique:0
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=

#program base.
v(X) :- e(X,_).
s(X) :- e(_,X).
k <= { sel(X): s(X) } <= 0.
cov(X) :- v(X); e(X,S); sel(S).
#false :- not cov(X); v(X).
