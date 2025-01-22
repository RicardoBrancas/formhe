%formhe-problem:A
%formhe-timestamp:1698329231

vertex(V) :- e(V, _).
vertex(V) :- e(_, V).
1 { assign(V, 1..k) } 1 :- vertex(V).
:- e(U, V), assign(U, C), assign(V, C).