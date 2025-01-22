%formhe-problem:E
%formhe-timestamp:1698339288
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(A, B), set(A, X), set(B, X).

1 { set(X, a); set(X, b) } 1 :- vertex(X).
edge(0, B) :- edge(b, B), set(b, X), set(B, I1), vertex(B).