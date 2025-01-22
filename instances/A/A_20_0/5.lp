%formhe-problem:A
%formhe-timestamp:1698332121
%formhe-selfeval-lines:0 3 4
%formhe-selfeval-fix: :- e(V1, V2), assign(V1, C), assign(V2, C). vertex(V) :- e(U, V). 1 { assign(V, C) : color(C) } 1 :- vertex(V).

 :- e(V, V2), assign(V1, C), assign(V2, C).
vertex(U) :- e(U, V).
color((1..k)).
 :- e(U, V).
 :- color(k).