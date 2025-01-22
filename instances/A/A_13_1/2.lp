%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(_, V).

 :- e(U, V), assign(U, C), assign(V, C).
1 { assign(V, (0..k)) } 1 :- vertex(V).
vertex(V) :- e(V, _).
vertex(V) :- e(_, V), -assign(C, V).