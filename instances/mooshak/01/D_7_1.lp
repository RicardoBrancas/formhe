%formhe-problem:D
%formhe-timestamp:1667735014
%formhe-selfeval-lines:2
%formhe-selfeval-fix:0{sel(X) : vertex(X)}k.

vertex(X) :- e(_,X).
vertex(X) :- e(X,_).

k{sel(X) : vertex(X)}k.

cover(X,Y) :- sel(X), e(X, Y).
cover(X,Y) :- sel(Y), e(X, Y).  

:- not cover(X,Y), e(X,Y).