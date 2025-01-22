%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:nodes(X) :- e(X, _). nodes(Y) :- e(_, Y).

1 { sel(N) : nodes(N) } k.
nodes(X) :- e(X, _), e(I0; Y, (1..k)).
nodes(0) :- e(_, Y).
 :- e(X, Y), not sel(X), not sel(Y).