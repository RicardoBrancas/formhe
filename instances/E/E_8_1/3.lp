%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:set(A, b) :- vertex(A), not set(A, a). set(A, a) :- vertex(A), not set(A, b).

 :- set(V, a), edge(V, V2), not set(V2, b).
set(A, b) :- vertex(A), not set(A, a), edge((a..0), 0; 0).
 :- set(V, b), edge(V, V2), not set(V2, a).
set(A, a) :- vertex(A), -edge(A, b).