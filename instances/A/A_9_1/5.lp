%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:vertex(V) :- e(_, V).  :- e(V1, V2), assign(V1, C), assign(V2, C). vertex(V) :- e(V, _).

vertex(V) :- e(C, V).
 :- assign(V1, 1), assign(V1, C), assign(V2, C).
1 { assign(V, (1..k)) } 1 :- vertex(V).
vertex(k) :- e(0, _).