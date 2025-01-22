%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- e(U, V), assign(U, C), assign(V, C). vertex(V) :- e(_, V). 1 { assign(V, (1..k)) } 1 :- vertex(V).

vertex(V) :- e(V, _).
 :- e(C, V), assign(U, C), assign(V, C), C == 0.
vertex(V) :- e(0, V).
1 { assign(V, (0..k)) } V :- vertex(V).