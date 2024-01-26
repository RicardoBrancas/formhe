%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { set(NODE, GROUP) : group(GROUP) } 1 :- vertex(NODE).

group(a).
group(b).

set(NODE, GROUP) :- vertex(NODE), group(GROUP).

:- set(N, G), set(N1, G), N1 != N, edge(N, N1).