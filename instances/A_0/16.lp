%formhe-groundtruth:../../correct_instances/mooshak/A_0_1.lp
%formhe-selfeval-lines:2 4
%formhe-selfeval-deleted-lines:0
%formhe-selfeval-changes-generate:2
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:4
%formhe-selfeval-changes-test-n:1
%formhe-selfeval-changes-test-n-unique:1
%formhe-domain-predicates:assign/2
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShhLCBkKS4KZShjLCBkKS4K
%formhe-instance-base64:I2NvbnN0IGsgPSAzLgoKZShhLCBiKS4KZShhLCBjKS4KZShiLCBjKS4KZShjLCBkKS4KZShjLCBlKS4KZShjLCBmKS4KZShjLCBnKS4=

#program base.
color((1..k)).
node(N) :- e(_,N).
node(N) :- assign(N,_).
1 <= { assign(N,C): color(C) } <= 1 :- node(N).
#false :- e(N,M); e(N,C); assign(M,C).
