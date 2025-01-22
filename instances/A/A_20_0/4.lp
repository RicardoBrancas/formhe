%formhe-problem:A
%formhe-timestamp:1698332121
%formhe-selfeval-lines:3
%formhe-selfeval-fix:vertex(U) :- e(U, V).

color((1..k)).
 :- e(V1, V2), assign(V1, C), assign(V2, C).
vertex(V) :- e(U, V).
 :- e(U, V).
1 { assign(V, C) : color(C) } 1 :- vertex(V).