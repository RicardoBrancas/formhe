%formhe-groundtruth:../../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-instance-base64:dmVydGV4KGEpLiB2ZXJ0ZXgoYikuIHZlcnRleChjKS4gdmVydGV4KGUpLiB2ZXJ0ZXgoMSkuIHZlcnRleCgyKS4gdmVydGV4KDMpLiB2ZXJ0ZXgoNCkuIHZlcnRleCg1KS4KZWRnZShhLDEpLiBlZGdlKGEsMikuIGVkZ2UoYiwzKS4gZWRnZShjLDIpLiBlZGdlKGMsNCkuIGVkZ2UoZSwxKS4gZWRnZShlLDQpLiBlZGdlKGUsNSku
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:edge(Y, X) :- edge(X, Y).  :- edge(X, Y), set(X, S), set(Y, S).

edge(Y, X) :- edge(X, Y), edge(_, _).
1 { set(X, a); set(X, b) } 1 :- vertex(X).
edge(Y, Y) :- edge(X, Y), set(a, S), set(_, S).