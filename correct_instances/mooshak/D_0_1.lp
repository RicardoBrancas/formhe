vx(X) :- e(X,Y).
vx(X) :- e(Y,X).
0 { sel(X) : vx(X) } k.
:- not sel(X), not sel(Y), e(X,Y).