%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:set(A, a) | set(A, b) :- edge(A, _).  :- edge(A, B), set(A, b), set(B, b).

 :- edge(A, B), set(A, a), set(B, a).
set(A, b) | set(A, b) :- edge(A, _), edge((1..0), 0; 1).
set(A, a) | set(A, b) :- edge(_, A).
 :- edge(A, B), edge(A, a), B == b.