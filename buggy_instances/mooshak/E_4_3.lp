%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:1 { set(V, G): g(G) } 1 :- e(V, _) . 1 { set(V, G): g(G) } 1 :- e(_, V) . e(A,B) :- edge(A,B).

% % Mooshak-exercises in lab13 from ALC
% % Problem E

% % facts
% %% vertexes
% v(a). v(b). v(c). 
% v(d). v(e).

% %% edges 
% e(a,b). e(b,c). e(c,d).
% e(d,a). e(d,e).

%% sets
g(a; b) .

% generator
{ set(V, g(G)): g(G) } 1 :- e(V, _) .
{ set(V, g(G)): g(G) } 1 :- e(_, V) .


% rule
set(V, a) :- v(V), not set(V, b) .
set(V, b) :- v(V), not set(V, a) .

% tester
%% we cannot have two sets of vertexes 
%%  such that any vertexes have an
%%  edge between them
:- e(V1, V2), set(V1, G), set(V2, G), V1 != V2 .

% #show set/2.