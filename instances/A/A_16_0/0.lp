%formhe-problem:A
%formhe-timestamp:1698330474
%formhe-selfeval-lines:4
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C).

vertex(Y) :- e(X, Y).
1 { assign(V, C) : color(C) } 1 :- vertex(V).
color((1..k)).
vertex(X) :- e(X, Y).
 :- e(X, Y), assign(X, C), assign(C, C).