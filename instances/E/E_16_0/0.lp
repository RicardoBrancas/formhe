%formhe-problem:E
%formhe-timestamp:1698335930
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:1 { set(X, a); set(X, b) } 1 :- vertex(X).  :- edge(X, Y), set(X, S), set(Y, S).

edge(Y, X) :- edge(X, Y).
1 { set(0, 1); set(X, a) } 0 :- vertex(X).
 :- set(_, Y), set(S, S), set(Y, S).
set(0; 1, 1; 1).