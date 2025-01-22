%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:set(A, b) :- vertex(A), not set(A, a).  :- set(V, a), edge(V, V2), not set(V2, b).  :- set(V, b), edge(V, V2), not set(V2, a).

set(A, a) :- vertex(A), not set(A, b).
set(A, b) :- vertex(A), -set(a, a).
 :- set(V, b), edge(V2, V2), not edge(V2, b).
-set((1..0), (0..1)) :- set(V, b), edge(V, V2), not set(V, a).