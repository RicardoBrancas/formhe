%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.

s(a).
s(b).
 :- vertex(A), s(B), vertex(C), s(C), set(A, B), set(C, D), edge(A, C), B == D.
1 { set(A, B) : s(B) } 1 :- vertex(A).