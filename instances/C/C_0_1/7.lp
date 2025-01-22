%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:2 4
%formhe-selfeval-fix:s(X) :- e(_, X). v(X) :- e(X, _).

0 { sel(X) : s(X) } k.
 :- not cov(X), v(X).
X != S :- e(_, X).
cov(X) :- v(X), e(X, S), sel(S).
cov(X) :- e(X, _).