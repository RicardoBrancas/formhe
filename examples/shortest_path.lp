% generate
0 { selected(X,Y) } 1 :- connected(X, Y).

% define
edge(1, 2).
edge(2, 3).
edge(2, 5).
edge(3, 4).
edge(5, 4).

connected(X, Y) :- edge(X, Y).
connected(X, Y) :- edge(Y, X).

path(X, Y) :- selected(X, Y).
path(X, Y) :- path(X, Z), path(Z, Y).

% test
:- start(X), end(Y), not path(X, Y).

cost(C) :- C = #count { X, Y: selected(X, Y)}.

#minimize {C : cost(C)}.

start(1).
end(3).