%formhe-problem:A
%formhe-timestamp:1668363858
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:1 { assign(A, (1..k)) } 1 :- e(A, B).  :- e(A, B), assign(A, C), assign(B, C).

1 { assign(A, 1; k) }  :- e(A, B).
1 { assign(A, (1..k)) } 1 :- e(B, A).
0 == 1 :- e(A, B), assign(I1, A), assign(C, C), k == 0.