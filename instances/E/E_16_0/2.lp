%formhe-problem:E
%formhe-timestamp:1698335930
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- edge(X, Y), set(X, S), set(Y, S). 1 { set(X, a); set(X, b) } 1 :- vertex(X).

edge(Y, X) :- edge(X, Y).
 :- X == Y, set(S, b), set(Y, S), edge(Y, Y).
vertex(1) :- vertex(X).