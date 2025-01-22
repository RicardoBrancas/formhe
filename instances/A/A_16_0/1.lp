%formhe-problem:A
%formhe-timestamp:1698330474
%formhe-selfeval-lines:4
%formhe-selfeval-fix: :- e(X, Y), assign(X, C), assign(Y, C).

vertex(Y) :- e(X, Y).
color((1..k)).
1 { assign(V, C) : color(C) } 1 :- vertex(V).
vertex(X) :- e(X, Y).
 :- e(X, I1), assign(_, C), assign(Y; X, C).