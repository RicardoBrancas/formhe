%formhe-problem:D
%formhe-timestamp:1698335743
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { sel(X) : vertex(X)} k.

vertex(X) :- e(X, Y).
vertex(Y) :- e(X, Y).

k { sel(X) : vertex(X)} k.

covered(X, Y) :- sel(X), e(X, Y).
covered(X, Y) :- sel(Y), e(X, Y).

:- not covered(X, Y), e(X, Y).

#show sel/1.