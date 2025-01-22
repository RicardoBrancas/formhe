%formhe-problem:A
%formhe-timestamp:1667930838
%formhe-selfeval-lines:3
%formhe-selfeval-fix:1 { assign(V, C) : color(C) } 1 :- v(V).

color((1..k)).
v(X) :- e(X, _).
v(X) :- e(_, X).
 { assign(V, C) : color(C) } 1 :- v(V).
 :- v(A), v(B), e(A, B), assign(A, C), assign(B, C).