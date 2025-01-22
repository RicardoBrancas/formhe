%formhe-problem:D
%formhe-timestamp:1699276601
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- e(X, Y), not sel(X), not sel(Y). nodes(Y) :- e(_, Y).

nodes(X) :- e(X, _).
1 { sel(N) : nodes(N) } k.
 :- e(N, Y), -sel(X), not nodes(Y).
 :- e(_, Y), e(I1; k, (0..1)).