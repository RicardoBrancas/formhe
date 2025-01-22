%formhe-problem:A
%formhe-timestamp:1667731437
%formhe-selfeval-lines:2 4
%formhe-selfeval-fix:1 { assign(X, C) : color(C) } 1 :- vertex(X). color((1..k)).  :- e(Y, X), assign(X, C), assign(Y, C), color(C).

vertex(X) :- e(_, X).
 :- e(X, Y), assign(X, C), assign(Y, C), color(C).
1 { assign(X, C) : color(C) } 1 :- vertex(X), e((X..k); (X..k), (X..k); (X..k)).
vertex(X) :- e(X, _).
 :- Y != X, assign(X, C), assign(Y, C), color(C).