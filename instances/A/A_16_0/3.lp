%formhe-problem:A
%formhe-timestamp:1698330474
%formhe-selfeval-lines:1 3 4
%formhe-selfeval-fix:vertex(X) :- e(X, Y). 1 { assign(V, C) : color(C) } 1 :- vertex(V).  :- e(X, Y), assign(X, C), assign(Y, C).

color((1..k)).
 :- e(X, Y).
vertex(Y) :- e(X, Y).
1 { assign(V, C) : vertex(C) } 1 :- vertex(V).
 :- X != Y, e(X, 1), assign(Y, C).