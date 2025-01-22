%formhe-problem:A
%formhe-timestamp:1668363858
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:1 { assign(A, (1..k)) } 1 :- e(B, A). 1 { assign(A, (1..k)) } 1 :- e(A, B).

 :- e(B, _).
 :- e(A, B), assign(A, C), assign(B, C).
1 { e(A, (1..k)) } 1 :- assign(A, B).
 :- assign(1; 0, (0..1)).