s(X) :- e(X, E).

k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E), X != Y.

:- inter(X, Y), sel(X), sel(Y).

%a