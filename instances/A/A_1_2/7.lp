%formhe-problem:A
%formhe-timestamp:1668363858
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix: :- e(A, B), assign(A, C), assign(B, C). 1 { assign(A, (1..k)) } 1 :- e(B, A).

 :- e(A, B), assign(C, _), B != C.
1 { assign(A, (1..k)) } 1 :- e(A, B).
1 { assign(A, 0; k) }  :- e(B, A).
 :- 1 == 0.