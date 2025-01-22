%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:1 2 4 5
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C), color(C). 1 { assign(X, C) : color(C) } 1 :- vertex(X). color((1..k)).  :- e(Y, X), assign(X, C), assign(Y, C), color(C).

vertex(X) :- e(_, X).
 :- e(X, Y), assign(X, Y), assign(Y, C), color(C).
1 { assign(X, C) : color(C) } 1 :- color(X).
vertex(X) :- e(X, _).
color((1..0)).
 :- e(Y, I1), assign(X, C), assign(C, C), vertex(C).