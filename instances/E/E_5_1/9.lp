%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D. 1 { set(A, B) : s(B) } 1 :- vertex(A).

s(a).
s(b).
 :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == C.
1 { set(A, B) : vertex(B) } 1 :- vertex(A).