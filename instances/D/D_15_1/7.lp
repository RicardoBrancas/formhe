%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:1 3 4
%formhe-selfeval-fix:nodes(Y) :- e(_, Y). nodes(X) :- e(X, _).

1 { sel(N) : nodes(N) } k.
nodes(Y) :- e(1, Y).
 :- e(X, Y), not sel(X), not sel(Y).
e(X, (0..1)) :- e(X, _), -e(_, _).
 :- e((1..0), (1..0)).