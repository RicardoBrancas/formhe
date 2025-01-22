%formhe-problem:C
%formhe-timestamp:1667930683

element(E) :- e(E, X).
set(X) :- e(E, X).

:- sel(X), sel(Y), not e(E, X), not e(E, Y), X!= Y, set(X), set(Y), element(E).


k{sel(X): set(X)}k.

#show sel/1.