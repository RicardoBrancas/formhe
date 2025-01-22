%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- e(U, V), assign(U, C), assign(V, C).

vertex(V) :- e(V, _).
vertex(V) :- e(_, V).
1 { assign(V, (1..k)) } 1 :- vertex(V).
 :- e(C, V), assign(U, C), assign(V, C).