%formhe-groundtruth:../../mooshak/instances/2021_1.lp
%formhe-domain-predicates:scc/1
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGEpLiBlZGdlKGMsZCkuCmVkZ2UoZCxhKS4gZWRnZShlLGQpLiBlZGdlKGUsYykuCg==
%formhe-optimization-problem:True
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:reach(R, R) :- vertex(R). reach(R1, R2) :- edge(R1, R2), vertex(R1), vertex(R2). reach(R1, R3) :- edge(R1, R2), reach(R2, R3), vertex(R3). 0 { scc(R) } 1 :- vertex(R). :- scc(R1), scc(R2), not reach(R1, R2). #maximize { 1,R  : scc(R),vertex(R) } .

incoming(X) :- edge(Y, X), vertex(X), vertex(Y).
outgoing(X) :- edge(X, Y), vertex(X), vertex(Y).
scc(X) :- incoming(X), outgoing(X), vertex(X).

#show scc/1.