%formhe-problem:A
%formhe-timestamp:1698329230
%formhe-selfeval-lines:0 1 3
%formhe-selfeval-fix:1 { assign(V, (1..k)) } 1 :- vertex(V). vertex(V) :- e(_, V).  :- e(U, V), assign(U, C), assign(V, C).

0 { assign(V, 1; k) } 1 :- vertex(V).
 :- e(_, U).
vertex(V) :- e(V, _).
 :- U == V, assign(U, C), e(U, C), e(_, _).