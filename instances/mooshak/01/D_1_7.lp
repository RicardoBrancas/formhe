%formhe-problem:D
%formhe-timestamp:1668373938
%formhe-selfeval-lines:2
%formhe-selfeval-fix:0{sel(A): aux(A)}k.

aux(A) :- e(A,B).
aux(A) :- e(B,A).

k{sel(A): aux(A)}k.
:- not sel(A), not sel(B), e(A,B).