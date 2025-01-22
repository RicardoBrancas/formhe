%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:
%formhe-selfeval-fix:v(X) :- e(X, _).

0 { sel(X) : s(X) } k.
 :- not cov(X), v(X).
cov(X) :- v(X), e(X, S), sel(S).
s(X) :- e(_, X).