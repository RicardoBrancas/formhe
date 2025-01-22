%formhe-problem:E
%formhe-timestamp:1668374442
%formhe-selfeval-lines:3
%formhe-selfeval-fix:set(A, a) | set(A, b) :- edge(_, A).

 :- edge(A, B), set(A, b), set(B, b).
 :- edge(A, B), set(A, a), set(B, a).
set(A, a) | set(A, b) :- edge(A, _).
set(A, a) | set(A, b) :- edge(_, A), edge(a; a; a; a, a; a; a; a).