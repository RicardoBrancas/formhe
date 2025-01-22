%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:4
%formhe-selfeval-fix:s(X) :- e(_, X).

k { sel(X) : s(X) } k.
i(X) :- e(X, _).
cov(I) :- sel(X), e(I, X), i(I).
 :- not cov(X), i(X).
i(X) :- e(_, X).