%formhe-problem:C
%formhe-timestamp:1667930992
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:i(X) :- e(X, _). s(X) :- e(_, X). k { sel(X) : s(X) } k. cov(I) :- sel(X), e(I, X), i(I).

not cov(k) :- e(_, _).
s(X) :- e(_, X), e(k; X, 1; I).
cov(I) :- sel(X), e(I, 0), s(I).
 :- not cov(X), i(X).