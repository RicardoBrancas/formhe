%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:
%formhe-selfeval-fix:set(A, a) | set(A, b) :- edge(A, _).  :- edge(A, B), set(A, a), set(B, a).

set(A, a) | set(A, b) :- edge(_, A).
 :- edge(A, B), set(A, b), set(B, b).