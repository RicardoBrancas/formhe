%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K

group(a).
group(b).

:- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X!= Y.
:- set(X, Z), set(X, W), Z!= W, group(Z), group(W).

1{set(X, Z): vertex(X), group(Z)}1 :- vertex(X).

#show set/2.