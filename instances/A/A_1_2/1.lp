%formhe-problem:A
%formhe-timestamp:1668363858
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:1 { assign(A, (1..k)) } 1 :- e(A, B). 1 { assign(A, (1..k)) } 1 :- e(B, A).

 :- e(A, B), assign(A, C), assign(B, C).
1 { assign(A, 0; k) } 1 :- e(A, B).
1 { assign(A, 0; k) } 1 :- e(B, A).
 :- k == B.