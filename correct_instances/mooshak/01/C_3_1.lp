%formhe-problem:C
%formhe-timestamp:1667675683

cover(X) :- e(X,Y), sel(Y).
{sel(X)} :- e(Y,X).
total(S) :- S=#count{X : e(Y,X), sel(X)}.
:- total(S), S!=k.
:- e(X,Y), not cover(X).