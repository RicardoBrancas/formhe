%formhe-groundtruth:../../correct_instances/mooshak/A_0_1.lp
%formhe-selfeval-lines:2
%formhe-selfeval-deleted-lines:1
%formhe-selfeval-changes-generate:2
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:
%formhe-selfeval-changes-test-n:0
%formhe-selfeval-changes-test-n-unique:0
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=

#program base.
color((1..k)).
node(N) :- e(_,N).
%
k <= { assign(N,C): color(C) } <= 1 :- node(N).
#false :- e(N,M); assign(N,C); assign(M,C).
