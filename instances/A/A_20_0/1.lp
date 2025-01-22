%formhe-problem:A
%formhe-timestamp:1698332121
%formhe-selfeval-lines:0 4
%formhe-selfeval-fix: :- e(V1, V2), assign(V1, C), assign(V2, C). 1 { assign(V, C) : color(C) } 1 :- vertex(V).

 :- e(V1, V), assign(V1, C), assign(k, C).
vertex(U) :- e(U, V).
color((1..k)).
vertex(V) :- e(U, V).
1 { assign(k, C) : vertex(C) } 1 :- color(0).