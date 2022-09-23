%formhe-groundtruth:../../examples/shortest_path.lp
%formhe-mcs-query:path(1,2). path(2,3). path(1,3).

% generate
0 { selected(X,Y) } 1 :- connected(X, Y).

formhe_definition_begin.

edge(1, 2).
edge(2, 3).
edge(2, 5).
edge(3, 4).
edge(5, 4).

start(1).
end(3).

formhe_definition_end.

connected(X, Y) :- edge(X, Y).
connected(X, Y) :- edge(Y, X).

path(X, Y) :- selected(X, Y).
path(X, Y) :- path(X, Z), path(Z, Y).

% test
:- start(X), end(Y), -path(X, Y).

cost(C) :- C = #count { X, Y: selected(X, Y)}.

#minimize {C : cost(C)}.

