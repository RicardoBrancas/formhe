%formhe-problem:D
%formhe-timestamp:1667931042
%formhe-selfeval-lines:2
%formhe-selfeval-fix:0{sel(X): v(X)}k.

v(X) :- e(X,_).
v(X) :- e(_,X).

k{sel(X): v(X)}k.

:- not sel(X), not sel(Y), e(X,Y).

#show sel/1.