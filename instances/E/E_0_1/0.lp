%formhe-problem:E
%formhe-timestamp:1
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(X, Y), set(X, S), set(Y, S).

1 { set(X, a); set(X, b) } 1 :- vertex(X).
 :- X != Y, set(X, S), set(Y, S).