%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:
%formhe-selfeval-fix:s(a). s(b).

1 { set(A, B) : s(B) } 1 :- vertex(A).
 :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.