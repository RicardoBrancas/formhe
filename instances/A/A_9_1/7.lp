%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:vertex(V) :- e(_, V). 1 { assign(V, (1..k)) } 1 :- vertex(V).  :- e(V1, V2), assign(V1, C), assign(V2, C).

vertex(V) :- e(V, _).
vertex(V) :- e(C, V).
1 { assign(V, (1..k)) } 0 :- vertex(V).
e((1..k), C) :- e(V1, V2), assign(I0, V2), e(1, C), e(0; 0, (1..0)).