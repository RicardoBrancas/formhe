%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K

% vertexes
%vertex(a). vertex(b). vertex(c). vertex(d). vertex(e).
% edges
%edge(a,b). edge(b,c). edge(c,d). edge(d,a). edge(d,e).


set(A, a) :- vertex(A), not set(A, b).
set(A, b) :- vertex(A), not set(A, a).




:- set(V, a), edge(V, V2), not set(V2, b).

:- set(V, b), edge(V, V2), not set(V2, a).