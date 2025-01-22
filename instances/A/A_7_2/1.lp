%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- e(Y, X), assign(X, C), assign(Y, C), color(C).

1 { assign(X, C) : color(C) } 1 :- vertex(X).
 :- e(Y, X), assign(X, C), Y != C, color(C).
 :- e(X, Y), assign(X, C), assign(Y, C), color(C).
color((1..k)).
vertex(X) :- e(X, _).
vertex(X) :- e(_, X).