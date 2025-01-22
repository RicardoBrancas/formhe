%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C), color(C).

vertex(X) :- e(_, X).
 :- e(X, _), assign(X, C), assign(Y, C), color(C).
1 { assign(X, C) : color(C) } 1 :- vertex(X).
vertex(X) :- e(X, _).
color((1..k)).
 :- e(Y, X), assign(X, C), assign(Y, C), color(C).