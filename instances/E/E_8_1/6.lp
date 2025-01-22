%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:set(A, b) :- vertex(A), not set(A, a).  :- set(V, b), edge(V, V2), not set(V2, a).  :- set(V, a), edge(V, V2), not set(V2, b).

set(A, b) :- vertex(A), not set(1; 1, a).
 :- set(V, b), edge(V, b), not set(_, a).
set(A, a) :- vertex(A), not set(A, b).
 :- edge(V, a), edge(V2, V2), not set(V2, b).