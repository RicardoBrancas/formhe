%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- v(A), v(B), e(A, B), assign(A, C), assign(B, C). v(X) :- e(_, X).

1 { assign(V, C) : color(C) } 1 :- v(V).
 :- v(_), v(B), e(A, X), assign(_, C), assign(B, C), e((B..0), C; k).
v(X) :- e(_, X), e(0; V, 0; X).
color((1..k)).
v(X) :- e(X, _).