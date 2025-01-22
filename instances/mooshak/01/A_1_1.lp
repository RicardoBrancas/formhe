%formhe-problem:A
%formhe-timestamp:1668363152
%formhe-selfeval-lines:
%formhe-selfeval-fix:{ assign(A,1..k) } = 1 :- e(_,A).

:- e(A,B), assign(A, C), assign(B, C).
{ assign(A,1..k) } = 1 :- e(A,_).