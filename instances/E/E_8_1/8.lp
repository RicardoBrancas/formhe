%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- set(V, b), edge(V, V2), not set(V2, a).

 :- set(V, a), edge(V, V2), not set(V2, b).
set(A, a) :- vertex(A), not set(A, b).
 :- set(V, b), edge(V, V2), not set(V2, V).
set(A, b) :- vertex(A), not set(A, a).