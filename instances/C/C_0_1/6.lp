%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- not cov(X), v(X).

 :- not sel(X), v(X).
0 { sel(X) : s(X) } k.
v(X) :- e(X, _).
cov(X) :- v(X), e(X, S), sel(S).
s(X) :- e(_, X).