%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:set(A, a) :- vertex(A), not set(A, b).  :- set(V, a), edge(V, V2), not set(V2, b).

set(A, b) :- vertex(A), not set(A, a).
 :- set(V, b), edge(V, V2), not set(V2, a).
 :- vertex(A), not set(A, a).
 :- set(V2, a), edge(V, V2), not edge(V2, b).