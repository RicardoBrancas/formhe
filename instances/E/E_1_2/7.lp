%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- edge(A, B), set(A, a), set(B, a).

 :- edge(I1, B), set(A, a), set(B, a).
set(A, a) | set(A, b) :- edge(A, _).
set(A, a) | set(A, b) :- edge(_, A).
 :- edge(A, B), set(A, b), set(B, b).