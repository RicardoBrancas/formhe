%formhe-problem:E
%formhe-timestamp:1667931097
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- vertex(A), s(B), vertex(C), s(D), set(A, B), set(C, D), edge(A, C), B == D. s(a).

 :- vertex(A), s(B), vertex(C), s(D), set(_, B), set(C, D), edge(A, C), set(A, D).
1 { set(A, B) : s(B) } 1 :- vertex(A).
s(a) :- -edge(_, _).
s(b).