%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- set(V, a), edge(V, V2), not set(V2, b). set(A, b) :- vertex(A), not set(A, a).  :- set(V, b), edge(V, V2), not set(V2, a).

 :- V == a, edge(V, V2), not set(V2, b).
set(A, b) :- vertex(A), -set(A, a).
 :- set(V, b), edge(V, V2), -set(V2, a).
set(A, a) :- vertex(A), not set(A, b).