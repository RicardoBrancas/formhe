%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:cov(I) :- sel(X), e(I, X), i(I).  :- not cov(X), i(X). k { sel(X) : s(X) } k.

cov(I) :- sel(X), e(I, X), s(X).
 :- not i(X), cov(X).
k { sel(X) : s(X) } 1.
s(X) :- e(_, X).
i(X) :- e(X, _).