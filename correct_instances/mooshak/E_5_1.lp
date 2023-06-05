%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K

%set como definir este set
%2{s(A): vertex(A)}2.
s(a). s(b).

1{set(A,B): s(B) }1:- vertex(A).

:- vertex(A), s(B), vertex(C), s(D), set(A,B), set(C,D), edge(A,C), B==D.

#show set/2.