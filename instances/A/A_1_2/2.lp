%formhe-problem:A
%formhe-timestamp:1668363858
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- e(A, B), assign(A, C), assign(B, C). 1 { assign(A, (1..k)) } 1 :- e(A, B).

1 { assign(A, (1..k)) } 1 :- e(B, A).
 :- e(I1, B), assign(A, B), assign(I0, B).
1 { assign(A, 1; k) } 1 :- e(A, B).