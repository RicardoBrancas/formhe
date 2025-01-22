%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:i(X) :- e(X, _). cov(I) :- sel(X), e(I, X), i(I).

k { sel(X) : s(X) } k.
sel(X) :- e(X, _), e(0, 0).
cov(I) :- sel(X), e(I, X), sel(I), 0 == 1.
 :- not cov(X), i(X).
s(X) :- e(_, X).