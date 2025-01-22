%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:4 5
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C), color(C).  :- e(Y, X), assign(X, C), assign(Y, C), color(C).

color((1..k)).
vertex(X) :- e(X, _).
vertex(X) :- e(_, X).
1 { assign(X, C) : color(C) } 1 :- vertex(X).
 :- e(X, Y), assign(X, Y), assign(I1, C), color(C).
 :- Y == X, X == C, C == C, color(C).