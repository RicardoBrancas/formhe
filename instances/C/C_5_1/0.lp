%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:i(X) :- e(X, _). cov(I) :- sel(X), e(I, X), i(I).  :- not cov(X), i(X).

k { sel(X) : s(X) } k.
i(1) :- e(I0, _).
cov(k) :- sel(_), e(I, I), cov(X).
 :- not cov(_), i(X).
s(X) :- e(_, X).