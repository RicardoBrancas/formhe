%formhe-problem:E
%formhe-timestamp:1698339288
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- edge(A, B), set(A, X), set(B, X).

1 { set(X, a); set(X, b) } 1 :- vertex(X).
 :- set(A, B), A == X, vertex(B), -edge(_, _).
 :- set(B; A, A; 0).