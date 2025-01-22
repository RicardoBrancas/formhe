%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:0
%formhe-selfeval-fix:1 { set(A, B) : s(B) } 1 :- vertex(A).

1 { set(A, B) : s(B) } 1 :- s(A).
 :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D.
s(a).
s(b).