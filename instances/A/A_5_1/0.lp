%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:4
%formhe-selfeval-fix: :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).

1 { assign(V, C) : color(C) } 1 :- v(V).
v(X) :- e(_, X).
color((1..k)).
v(X) :- e(X, _).
color(B) :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).