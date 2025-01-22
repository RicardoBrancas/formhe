%formhe-problem:A
%formhe-timestamp:1698328244
%formhe-selfeval-lines:(), (0, 1, 3)
%formhe-selfeval-fix:"edge(X, Y) :- e(X, Y).","vertex(V) :- e(V, _). vertex(V) :- e(_, V). :- e(U, V), assign(U, C), assign(V, C)."

vertex(V) :- edge(V, _).
vertex(V) :- edge(_, V).
1 { assign(V, 1..k) } 1 :- vertex(V).
:- edge(U, V), assign(U, C), assign(V, C).