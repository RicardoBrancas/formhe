%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix: :- edge(A, B), set(A, a), set(B, a). set(A, a) | set(A, b) :- edge(A, _).  :- edge(A, B), set(A, b), set(B, b).

 :- edge(A, B), set(b, a), set(B, a).
set(A, a) | set(A, b) :- edge(A, _), edge((1..0), 0; 1).
set(A, a) | set(A, b) :- edge(_, A).
 :- set(b, B), edge(A, a), edge(B, b).