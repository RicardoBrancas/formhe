%formhe-problem:E
%formhe-timestamp:1698335930
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- edge(X, Y), set(X, S), set(Y, S).

edge(Y, X) :- edge(X, Y).
1 { set(X, a); set(X, b) } 1 :- vertex(X).
 :- edge(X, Y), edge(X, S), set(Y, S).