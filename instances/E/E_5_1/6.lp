%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:s(a). s(b).  :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.

vertex(a).
1 { set(A, B) : s(B) } 1 :- vertex(A).
 :- vertex(A), vertex(B), vertex(C), s(D), A != B, set(C, D), edge(A, _), set(B, D).
 :- set(1; 1, 1; 1).