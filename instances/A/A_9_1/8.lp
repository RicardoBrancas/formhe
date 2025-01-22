%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- e(V1, V2), assign(V1, C), assign(V2, C). 1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(V, _).

vertex(V) :- e(_, V).
 :- V1 == V, assign(V1, C), assign(V2, C), e(_, C).
1 { assign(V, (k..0)) } k :- vertex(V).
vertex(V) :- assign(V, 0).