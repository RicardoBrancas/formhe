%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:vertex(V) :- e(_, V). vertex(V) :- e(V, _).  :- e(U, V), assign(U, C), assign(V, C).

1 { assign(V, (1..k)) } 1 :- vertex(V).
vertex(V) :- e(_, V), assign(0; 0, (k..0)).
vertex(V) :- assign(V, _).
not assign(U, k) :- e(U, V), assign(U, C), e(V, C).