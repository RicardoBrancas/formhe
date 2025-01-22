%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y). nodes(X) :- e(X, _).

1 { sel(N) : nodes(N) } k.
nodes(Y) :- e(_, Y).
 :- e(X, Y), not sel(X), not nodes(Y).
nodes(X) :- e(X, I0).