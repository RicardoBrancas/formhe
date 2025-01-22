%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:s(b).  :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D. 1 { set(A, B) : s(B) } 1 :- vertex(A).

s(a).
vertex(b).
 :- vertex(A), s(B), vertex(C), s(0), set(A, B), set(C, 0), edge(D, C), B == a.
1 { set(A, B) : vertex(B) }  :- vertex(A).