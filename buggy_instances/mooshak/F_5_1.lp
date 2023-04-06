%formhe-groundtruth:../../mooshak/instances/2021_1.lp
%formhe-domain-predicates:scc/1
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGEpLiBlZGdlKGMsZCkuCmVkZ2UoZCxhKS4gZWRnZShlLGQpLiBlZGdlKGUsYykuCg==
%formhe-optimization-problem:True

{scc(A): vertex(A)}.

path(X,Y):- scc(X), scc(Y), edge(X,Y).
path(X,Z):- path(X,Y), scc(Z), edge(Y,Z).

:- scc(A), scc(B) , not path(A,B).

#maximize{1,A: scc(A) }.

#show scc/1.