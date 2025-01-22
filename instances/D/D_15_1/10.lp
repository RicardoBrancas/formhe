%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:
%formhe-selfeval-fix:nodes(X) :- e(X, _).  :- e(X, Y), not sel(X), not sel(Y). nodes(Y) :- e(_, Y).

1 { sel(N) : nodes(N) } k.