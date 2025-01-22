%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix:1 { assign(V, C) : color(C) } 1 :- v(V).  :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).

1 { assign(V, C) : color(C) } 1 :- color(V).
color((1..k)).
v(X) :- e(_, X).
 :- v(I1), v(B), assign(A, B), assign(A, C), assign(B, k).
v(X) :- e(X, _).