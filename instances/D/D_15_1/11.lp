%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix:nodes(Y) :- e(_, Y). nodes(X) :- e(X, _).

1 { sel(N) : nodes(N) } k.
nodes(Y) :- e(_, Y), e(X; 0, X; k).
 :- e(X, Y), not sel(X), not sel(Y).
nodes(X) :- e(X, _), e(I0; 0, (1..0)).