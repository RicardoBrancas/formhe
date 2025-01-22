%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:
%formhe-selfeval-fix:cov(I) :- sel(X), e(I, X), i(I).

 :- not cov(X), i(X).
k { sel(X) : s(X) } k.
s(X) :- e(_, X).
i(X) :- e(X, _).