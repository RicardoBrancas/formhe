%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:vertex(V) :- e(_, V). vertex(V) :- e(V, _).

vertex(V) :- e(C, V).
 :- e(V1, V2), assign(V1, C), assign(V2, C).
1 { assign(V, (1..k)) } 1 :- vertex(V).
vertex(V) :- assign(V, _).