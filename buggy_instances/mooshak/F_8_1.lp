%formhe-groundtruth:../../mooshak/instances/2021_1.lp
%formhe-domain-predicates:scc/1
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGEpLiBlZGdlKGMsZCkuCmVkZ2UoZCxhKS4gZWRnZShlLGQpLiBlZGdlKGUsYykuCg==
%formhe-optimization-problem:True
%formhe-selfeval-lines:2
%formhe-selfeval-fix:0 { scc(R) } 1 :- vertex(R). :- scc(R1), scc(R2), not path(R1, R2). #maximize { 1,R  : scc(R),vertex(R) } .

% vertexes
%vertex(a). vertex(b). vertex(c). vertex(d). vertex(e).
% edges
%edge(a,b). edge(b,c). edge(c,a). edge(c,d). edge(d,a). edge(e,d). edge(e,c).


path(A,B) :- edge(A,B).
path(A, B) :- edge(A, C), path(C, B).


scc(A) :- path(V,A), path(A, V).



%:- scc(A), not path(A,B), scc(B).