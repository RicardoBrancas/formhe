#const k = 3.

formhe_definition_begin.

e(a, b).
e(a, c).
e(a, d).
e(b, c).
e(c, d).

formhe_definition_end.

color(1..k).

node(N) :- e(_, N).
node(N) :- e(N, _).

1 { assign(N, C) : color(C) } 1 :- node(N).

:- e(N, M), assign(N, C), assign(M, C).