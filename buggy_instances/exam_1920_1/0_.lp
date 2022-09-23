#const k = 2.

e(a, 1).
e(a, 2).
e(b, 3).
e(b, 4).
e(c, 1).
e(c, 3).

s(X) :- e(X, E).

k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E), X != Y.

:- inter(X, Y), s(X), s(Y).

%f