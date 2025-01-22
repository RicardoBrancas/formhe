%formhe-problem:E
%formhe-timestamp:1667770259

% vertexes
%vertex(a). vertex(b). vertex(c). vertex(d). vertex(e).
% edges
%edge(a,b). edge(b,c). edge(c,d). edge(d,a). edge(d,e).


set(A, a) :- vertex(A), not set(A, b).
set(A, b) :- vertex(A), not set(A, a).




:- set(V, a), edge(V, V2), not set(V2, b).

:- set(V, b), edge(V, V2), not set(V2, a).