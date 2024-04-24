%formhe-groundtruth:../01/E_0_1.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-instance-base64:dmVydGV4KGEpLiB2ZXJ0ZXgoYikuIHZlcnRleChjKS4gdmVydGV4KGUpLiB2ZXJ0ZXgoMSkuIHZlcnRleCgyKS4gdmVydGV4KDMpLiB2ZXJ0ZXgoNCkuIHZlcnRleCg1KS4KZWRnZShhLDEpLiBlZGdlKGEsMikuIGVkZ2UoYiwzKS4gZWRnZShjLDIpLiBlZGdlKGMsNCkuIGVkZ2UoZSwxKS4gZWRnZShlLDQpLiBlZGdlKGUsNSku

group(a).
group(b).
1{set(NODE, GROUP): vertex(NODE), group(GROUP)}1:-vertex(NODE).
:-set(N1,G), set(N2,G), edge(N1,N2).