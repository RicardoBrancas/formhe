%formhe-groundtruth:./E_0_1.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-instance-base64:dmVydGV4KGEpLiB2ZXJ0ZXgoYikuIHZlcnRleChjKS4gdmVydGV4KGUpLiB2ZXJ0ZXgoMSkuIHZlcnRleCgyKS4gdmVydGV4KDMpLiB2ZXJ0ZXgoNCkuIHZlcnRleCg1KS4KZWRnZShhLDEpLiBlZGdlKGEsMikuIGVkZ2UoYiwzKS4gZWRnZShjLDIpLiBlZGdlKGMsNCkuIGVkZ2UoZSwxKS4gZWRnZShlLDQpLiBlZGdlKGUsNSku

%set como definir este set
%2{s(A): vertex(A)}2.
s(a). s(b).

1{set(A,B): s(B) }1:- vertex(A).

:- vertex(A), s(B), vertex(C), s(D), set(A,B), set(C,D), edge(A,C), B==D.

#show set/2.