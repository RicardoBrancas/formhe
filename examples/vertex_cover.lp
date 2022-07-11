% generate
0 { cover(X) ; cover(Y) } 2 :- edge(X, Y).

% define
edge(1, 2).
edge(2, 3).
edge(2, 5).
edge(3, 4).
edge(5, 4).

% test
:- edge(X, Y), not cover(X), not cover(Y).

cost(C) :- C = #count {X : cover(X)}.

#minimize {C : cost(C)}.