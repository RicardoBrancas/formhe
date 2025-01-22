%formhe-problem:A
%formhe-timestamp:1668363859

:- e(A,B), assign(A, C), assign(B, C).
1 { assign(A,1..k)} 1 :- e(A,B).
1 { assign(A,1..k)} 1 :- e(B,A).