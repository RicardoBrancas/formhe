%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:nodes(Y) :- e(_, Y). nodes(X) :- e(X, _).

 :- e(X, Y), not sel(X), not sel(Y).
1 { sel(N) : nodes(N) } k.
nodes(Y) :- e(0, Y).
nodes(X) :- e(X, _), e(X; 0, k; 0).