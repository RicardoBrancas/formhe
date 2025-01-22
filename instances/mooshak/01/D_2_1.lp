%formhe-problem:D
%formhe-timestamp:1667931488
%formhe-selfeval-lines:3
%formhe-selfeval-fix:0{sel(X): vertex(X)}k.

vertex(X) :- e(X, Y).
vertex(Y) :- e(X, Y).


:- vertex(X), vertex(Y), not sel(X), not sel(Y), e(X, Y), X!=Y.


k{sel(X): vertex(X)}k.