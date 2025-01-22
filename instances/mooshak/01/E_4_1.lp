%formhe-problem:E
%formhe-timestamp:1667829090
%formhe-selfeval-lines:2 3 4
%formhe-selfeval-fix:1 { set(V, G): g(G) } 1 :- edge(V, _) . 1 { set(V, G): g(G) } 1 :- edge(_, V) . :- edge(V1, V2), set(V1, G1), set(V2, G2), G1 == G2 .

% Mooshak-exercises in lab13 from ALC
% Problem E

% facts
% %% vertexes
% v(a). v(b). v(c). 
% v(d). v(e).

% %% edges 
% e(a,b). e(b,c). e(c,d).
% e(d,a). e(d,e).

%% sets
g(a). g(b).

% generator
{ set(V, G): g(G) } 1 :- e(V, _) .
{ set(V, G): g(G) } 1 :- e(_, V) .

% tester
%% we cannot have two sets of vertexes 
%%  such that any vertexes have an
%%  edge between them
:- e(V1, V2), set(V1, G1), set(V2, G2), G1 != G2 .