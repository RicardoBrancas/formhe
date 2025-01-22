%formhe-problem:E
%formhe-timestamp:1698335930
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:edge(Y, X) :- edge(X, Y).  :- edge(X, Y), set(X, S), set(Y, S).

edge(a, a) :- set(X, _), edge((0..1), 0; 1).
1 { set(X, a); set(X, b) } 1 :- vertex(X).
 :- set(S, Y), set(X, a), set(Y, S).