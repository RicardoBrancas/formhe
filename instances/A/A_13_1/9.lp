%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:1 { assign(V, (1..k)) } 1 :- vertex(V).  :- e(U, V), assign(U, C), assign(V, C).

 :- vertex(V), not assign(_, _).
vertex(V) :- e(_, V).
vertex(V) :- e(V, _).
 :- e(U, V), assign(U, U), assign(V, C).