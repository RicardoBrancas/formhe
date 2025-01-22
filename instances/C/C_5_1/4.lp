%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:cov(I) :- sel(X), e(I, X), i(I).  :- not cov(X), i(X).

k { sel(X) : s(X) } k.
i(X) :- e(X, _).
cov(X) :- sel(X), e(I0, X), i(I), e((0..k), (0..k)).
 :- not cov(0), e(X, I1).
s(X) :- e(_, X).