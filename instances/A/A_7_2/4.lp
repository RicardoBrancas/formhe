%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:1 3 5
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C), color(C). color((1..k)). vertex(X) :- e(X, _).

 :- e(Y, X), assign(X, C), assign(Y, C), color(C).
 :- e(X, Y), assign(X, k), assign(Y, C), color(C).
vertex(X) :- e(_, X).
color((0..k)).
1 { assign(X, C) : color(C) } 1 :- vertex(X).
vertex(X) :- e(X, 1).