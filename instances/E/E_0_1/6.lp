%formhe-problem:E
%formhe-timestamp:1
%formhe-selfeval-lines:1
%formhe-selfeval-fix:1 { set(X, a); set(X, b) } 1 :- vertex(X).

 :- edge(X, Y), set(X, S), set(Y, S).
 { set(X, a); X == b } 1 :- vertex(X).