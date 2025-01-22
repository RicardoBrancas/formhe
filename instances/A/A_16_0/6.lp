%formhe-problem:A
%formhe-timestamp:1698330474
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:vertex(Y) :- e(X, Y). 1 { assign(V, C) : color(C) } 1 :- vertex(V). vertex(X) :- e(X, Y).

e(Y, (1..k)) :- e(X, Y).
1 { assign(V, C) : color(C) }  :- vertex(V).
color((1..k)).
vertex(Y) :- assign(X, Y).
 :- e(X, Y), assign(X, C), assign(Y, C).