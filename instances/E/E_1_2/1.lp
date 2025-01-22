%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- edge(A, B), set(A, b), set(B, b).  :- edge(A, B), set(A, a), set(B, a). set(A, a) | set(A, b) :- edge(A, _).

 :- a != B, set(A, b), set(B, b).
 :- a == B, set(I1, a), b == a.
set(A, a) | set(A, b) :- set(A, _).
set(A, a) | set(A, b) :- edge(_, A).