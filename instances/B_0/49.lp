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
s(X) :- e(X,E).
%
inter(X,Y) :- e(X,E); e(Y,E); X != Y.
#false :- inter(X,Y); sel(X); sel(Y).
