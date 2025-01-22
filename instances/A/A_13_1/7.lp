%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(V, _).  :- e(U, V), assign(U, C), assign(V, C).

 :- vertex(V).
vertex(V) :- e(_, V).
vertex(V) :- assign(V, _).
 :- e(U, I0), assign(U, C), 0 != C.