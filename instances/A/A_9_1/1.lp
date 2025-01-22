%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- e(V1, V2), assign(V1, C), assign(V2, C).

vertex(V) :- e(V, _).
vertex(V) :- e(_, V).
1 { assign(V, (1..k)) } 1 :- vertex(V).
 :- e(V1, C), assign(V1, C), assign(V2, C).