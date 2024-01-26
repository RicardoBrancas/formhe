%formhe-groundtruth:../../correct_instances/mooshak/D_0_1.lp
%formhe-selfeval-lines:1
%formhe-selfeval-deleted-lines:1
%formhe-selfeval-changes-generate:1
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:
%formhe-selfeval-changes-test-n:0
%formhe-selfeval-changes-test-n-unique:0
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCAyKS4KZSgxLCAzKS4KZSg0LCAzKS4KZSg0LCA1KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4KZSgyLCAzKS4KZSgzLCA0KS4KZSg0LCA1KS4KZSg1LCA2KS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAxLgoKZSgxLCAzKS4KZSgyLCAzKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZSgxLCAyKS4=

#program base.
vx(X) :- e(X,Y).
e(X) :- e(Y,X).
%
#false :- not sel(X); not sel(Y); e(X,Y).
