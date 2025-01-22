%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:1
%formhe-selfeval-fix:1 { sel(N) : nodes(N) } k.

 :- e(X, Y), not sel(X), not sel(Y).
k { sel(N) : nodes(N) } k.
nodes(Y) :- e(_, Y).
nodes(X) :- e(X, _).