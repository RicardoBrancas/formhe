%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:4
%formhe-selfeval-fix:v(X) :- e(X, _).

1 { assign(V, C) : color(C) } 1 :- v(V).
color((1..k)).
v(X) :- e(_, X).
 :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).
v(X) :- e(X, 0).