%formhe-groundtruth:../../mooshak/instances/2021_2.lp
%formhe-domain-predicates:set/2
%formhe-instance-base64:JSB2ZXJ0ZXhlcwp2ZXJ0ZXgoYSkuIHZlcnRleChiKS4gdmVydGV4KGMpLiB2ZXJ0ZXgoZCkuIHZlcnRleChlKS4KJSBlZGdlcwplZGdlKGEsYikuIGVkZ2UoYixjKS4gZWRnZShjLGQpLiBlZGdlKGQsYSkuIGVkZ2UoZCxlKS4K

% Mooshak-exercises in lab13 from ALC
% Problem E


%% sets
g(a; b) .

% generator
1 { set(V, G): g(G) } 1 :- vertex(V) .
1 { set(V, G): g(G) } 1 :- vertex(V) .


% rule
% set(V, a) :- vertex(V), not set(V, b) .
% set(V, b) :- vertex(V), not set(V, a) .

% tester
%% we cannot have two sets of vertexes 
%%  such that any vertexes have an
%%  edge between them
:- edge(V1, V2), set(V1, G), set(V2, G), V1 != V2 .

#show set/2.