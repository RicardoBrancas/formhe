%formhe-groundtruth:../../correct_instances/mooshak/B_0_1.lp
%formhe-selfeval-lines:1
%formhe-selfeval-deleted-lines:1
%formhe-selfeval-changes-generate:1
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:
%formhe-selfeval-changes-test-n:0
%formhe-selfeval-changes-test-n-unique:0
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

#program base.
%
k <= { sel(X): s(X) } <= k.
inter(X,Y) :- e(X,E); sel(Y,E); X != Y.
#false :- inter(X,Y); sel(X); sel(Y).
