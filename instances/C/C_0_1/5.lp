%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:v(X) :- e(X, _). s(X) :- e(_, X). 0 { sel(X) : s(X) } k.

v(X) :- e(X, _), sel(k; 1).
 :- e(_, X), e((k..1), 1; k).
0 { sel(X) : v(X) } k.
cov(X) :- v(X), e(X, S), sel(S).
 :- not cov(X), v(X).