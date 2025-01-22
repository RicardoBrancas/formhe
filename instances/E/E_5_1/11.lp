%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.

 :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(I1, D), edge(A, C), B == D.
1 { set(A, B) : s(B) } 1 :- vertex(A).
s(a).
s(b).