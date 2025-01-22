%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y).

1 { sel(N) : nodes(N) } k.
nodes(Y) :- e(_, Y).
 :- e(X, Y), not sel(Y), not sel(Y).
nodes(X) :- e(X, _).