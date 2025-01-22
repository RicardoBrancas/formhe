%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:2 3 4
%formhe-selfeval-fix:v(X) :- e(_, X). v(X) :- e(X, _). color((1..k)).

1 { assign(V, C) : color(C) } 1 :- v(V).
 :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).
color(X) :- e(_, X).
v(X) :- e(X, _), 0 == k.
color((0..k)).