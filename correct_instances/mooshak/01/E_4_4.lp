%formhe-problem:E
%formhe-timestamp:1667846436

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