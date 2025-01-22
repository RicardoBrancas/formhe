%formhe-problem:C
%formhe-timestamp:1667930993

i(X) :- e(X,_).
s(X) :- e(_, X).

k{sel(X): s(X)}k.

cov(I) :- sel(X), e(I, X), i(I).

:- not cov(X), i(X).

#show sel/1.