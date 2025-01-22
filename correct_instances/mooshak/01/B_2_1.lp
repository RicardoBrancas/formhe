%formhe-problem:B
%formhe-timestamp:1667929598

set(X) :- e(X, E).
joint(X, Y) :- e(X, E), e(Y, E), X != Y.
:- joint(X, Y), set(X), set(Y), sel(X), sel(Y).

k{sel(X): set(X)}k.