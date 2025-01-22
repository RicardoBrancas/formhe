%formhe-problem:E
%formhe-timestamp:1
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(X, Y), set(X, S), set(Y, S).

1 { set(X, a); set(X, b) } 1 :- vertex(X).
 :- set(I1, Y), set(_, a), set(Y, S), edge((1..0), (1..S)).