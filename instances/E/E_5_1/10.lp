%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.

1 { set(A, B) : s(B) } 1 :- vertex(A).
 :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), A == C, B == b.
s(a).
s(b).