%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:3
%formhe-selfeval-fix:vertex(X) :- e(X, _).

 :- e(Y, X), assign(X, C), assign(Y, C), color(C).
1 { assign(X, C) : color(C) } 1 :- vertex(X).
 :- e(X, Y), assign(X, C), assign(Y, C), color(C).
vertex(X) :- e(X, X).
color((1..k)).
vertex(X) :- e(_, X).