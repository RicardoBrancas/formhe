%formhe-groundtruth:../../correct_instances/mooshak/E_0_1.lp
%formhe-selfeval-lines:0 1
%formhe-selfeval-deleted-lines:0
%formhe-selfeval-changes-generate:0
%formhe-selfeval-changes-generate-n:1
%formhe-selfeval-changes-generate-n-unique:1
%formhe-selfeval-changes-test:1
%formhe-selfeval-changes-test-n:1
%formhe-selfeval-changes-test-n-unique:1
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K

#program base.
1 <= { set(X,a); edge(X,b) } <= 1 :- vertex(X).
#false :- set(X,Y); set(X,S); set(Y,S).
