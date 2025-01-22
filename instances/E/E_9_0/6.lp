%formhe-problem:E
%formhe-timestamp:1698339288
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(A, B), set(A, X), set(B, X).

1 { set(X, a); set(X, b) } 1 :- vertex(X).
0 != 1 :- edge(A, B), A == X, b == X.