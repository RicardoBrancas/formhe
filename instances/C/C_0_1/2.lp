%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:0 { sel(X) : s(X) } k. v(X) :- e(X, _). cov(X) :- v(X), e(X, S), sel(S).

 :- not cov(X), v(X).
0 { cov(X) : s(X) } k.
v(X) :- e(X, k), cov(X).
v(X) :- v(X), e(0, S), v(S).
s(X) :- e(_, X).