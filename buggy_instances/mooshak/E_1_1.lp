%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix::- edge(A,B), set(A, a), set(B, a). :- edge(A,B), set(A, b), set(B, b).

set(A,a) | set(A,b) :- edge(A,_).
set(A,a) | set(A,b) :- edge(_,A).


:- edge(A,B), set(A, a), set(B, b).
:- edge(A,B), set(A, b), set(B, a).