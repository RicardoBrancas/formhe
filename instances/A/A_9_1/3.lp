%formhe-problem:A
%formhe-timestamp:1698329276
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- e(V1, V2), assign(V1, C), assign(V2, C). 1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(V, _).

vertex(V) :- e(_, V).
 :- V1 == C, assign(C, C), assign(V2, C).
1 { assign(V, (1..0)) } V :- vertex(V).
vertex(V1) :- e(V1, _).