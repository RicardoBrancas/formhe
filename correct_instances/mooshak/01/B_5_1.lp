%formhe-problem:B
%formhe-timestamp:1667930929

s(X) :- e(X,I).

k{sel(X): s(X)}k.

inter(X,Y) :- e(X, E), e(Y, E), X!=Y.

:- inter(X,Y), sel(X), sel(Y).

#show sel/1.