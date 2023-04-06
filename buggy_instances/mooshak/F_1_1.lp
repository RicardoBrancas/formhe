%formhe-groundtruth:../../mooshak/instances/2021_1.lp
%formhe-domain-predicates:scc/1
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGEpLiBlZGdlKGMsZCkuCmVkZ2UoZCxhKS4gZWRnZShlLGQpLiBlZGdlKGUsYykuCg==
%formhe-optimization-problem:True
%formhe-selfeval-lines:3
%formhe-selfeval-fix:link(A,A) :- vertex(A). 0 { scc(R) } 1 :- vertex(R). :- scc(R1), scc(R2), not link(R1, R2). #maximize { 1,R  : scc(R),vertex(R) }.

link(A,B) :- edge(A,C), edge(C,B), A!=B.
link(A,B) :- edge(A,C), link(C,B), A!=B.
link(A,B) :- link(A,C), link(C,B), A!=B.
scc(A),scc(B) :- link(A,B), link(B,A), A!=B.