v(X) :- e(X, _).
s(X) :- e(_, X).
0 { sel(X) : s(X) } k.
cov(X) :- v(X), e(X, S), sel(S).
:- not cov(X), v(X).