%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:vertex(V) :- e(_, V).  :- e(V1, V2), assign(V1, C), assign(V2, C).

vertex(V) :- e(V, _).
vertex(V) :- e(V2, V).
1 { assign(V, (1..k)) } 1 :- vertex(V).
 :- V1 == V2, assign(V1, C), e(1, C), assign(k; 0, (C..0)).