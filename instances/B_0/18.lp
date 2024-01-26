%formhe-groundtruth:../../correct_instances/mooshak/B_0_1.lp
%formhe-selfeval-lines:0 3
%formhe-selfeval-deleted-lines:0
%formhe-selfeval-changes-generate:0
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:3
%formhe-selfeval-changes-test-n:1
%formhe-selfeval-changes-test-n-unique:1
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZShhLCAxKS4KZShhLCAyKS4KZShiLCAzKS4KZShiLCA0KS4KZShjLCAxKS4KZShjLCAzKS4K

#program base.
sel(X) :- e(X,E).
k <= { sel(X): s(X) } <= k.
inter(X,Y) :- e(X,E); e(Y,E); X != Y.
#false :- inter(X,Y); sel(X); s(Y).
