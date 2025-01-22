%formhe-problem:C
%formhe-timestamp:1667734195

set(X) :- e(_,X).
cover(N) :- sel(X), e(N, X).

k{sel(X) : set(X)}k.

:- not cover(N), e(N,_).