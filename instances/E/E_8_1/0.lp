%formhe-problem:E
%formhe-timestamp:1667770258
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix: :- set(V, b), edge(V, V2), not set(V2, a).  :- set(V, a), edge(V, V2), not set(V2, b). set(A, b) :- vertex(A), not set(A, a).

 :- V == b, edge(V, V2), not edge(V2, a).
 :- set(V, a), edge(_, V2), not set(V2, b), set(I1, _).
set(A, a) :- vertex(A), not set(A, b).
set(A, b) :- vertex(I0), -set(A, a).