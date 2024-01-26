%formhe-groundtruth:../../mooshak/instances/2021_1.lp
%formhe-domain-predicates:scc/1
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGEpLiBlZGdlKGMsZCkuCmVkZ2UoZCxhKS4gZWRnZShlLGQpLiBlZGdlKGUsYykuCg==
%formhe-optimization-problem:True

{scc(X):vertex(X)}.

reachable(X,X) :- vertex(X).
reachable(X,Y) :- edge(X,Y).
reachable(X,Y) :- reachable(X,Z), reachable(Z, Y).

#maximize{1, X : scc(X)}.

:- scc(X), scc(Y), not reachable(X,Y).