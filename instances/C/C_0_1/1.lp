%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:1 4
%formhe-selfeval-fix:cov(X) :- v(X), e(X, S), sel(S). s(X) :- e(_, X).

0 { sel(X) : s(X) } k.
cov(X) :- sel(X), e(X, S), sel(I1).
 :- not cov(X), v(X).
v(X) :- e(X, _).
s(X) :- e(_, X), 1 == 0.