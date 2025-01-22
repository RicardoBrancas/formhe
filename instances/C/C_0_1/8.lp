%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:2 3 4
%formhe-selfeval-fix:v(X) :- e(X, _). cov(X) :- v(X), e(X, S), sel(S). s(X) :- e(_, X).

0 { sel(X) : s(X) } k.
 :- not cov(X), v(X).
v(X) :- e(I0, _), s(X).
sel(X) :- v(X), e(X, S), v(I1).
s(k) :- e(k, X).