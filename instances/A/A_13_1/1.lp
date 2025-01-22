%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- e(U, V), assign(U, C), assign(V, C). 1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(V, _).

 :- e(U, V), assign(V, C), assign(V, C).
1 { assign(V, (0..k)) } 1 :- vertex(V).
 :- e(V, _).
vertex(V) :- e(_, V).