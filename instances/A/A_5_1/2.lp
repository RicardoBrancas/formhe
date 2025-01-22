%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix: :- v(A), v(B), e(A, B), assign(A, C), assign(B, C). v(X) :- e(_, X). color((1..k)).

1 { assign(V, C) : color(C) } 1 :- v(V).
 :- v(A), v(B), e(A, B), V == C, assign(B, C).
v(X) :- e(I0, X), assign(X, _).
color((1..0)).
v(X) :- e(X, _).