%formhe-groundtruth:../../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-instance-base64:dmVydGV4KGEpLiB2ZXJ0ZXgoYikuIHZlcnRleChjKS4gdmVydGV4KGUpLiB2ZXJ0ZXgoMSkuIHZlcnRleCgyKS4gdmVydGV4KDMpLiB2ZXJ0ZXgoNCkuIHZlcnRleCg1KS4KZWRnZShhLDEpLiBlZGdlKGEsMikuIGVkZ2UoYiwzKS4gZWRnZShjLDIpLiBlZGdlKGMsNCkuIGVkZ2UoZSwxKS4gZWRnZShlLDQpLiBlZGdlKGUsNSku
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:group(b).  :- set(N, G), set(N1, G), N1 != N, edge(N, N1).

group(a).
group(b) :- vertex(a).
1 { set(N, G) : group(G) } 1 :- vertex(N).
 :- set(N, G), N1 == G, N1 != N, set(N, N1).