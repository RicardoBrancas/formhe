%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:0
%formhe-selfeval-fix:0 { sel(X) : s(X) } k. cov(X) :- v(X), e(X, S), sel(S).

cov(X) :- cov(X), X == S, sel(I1).
 :- not cov(X), v(X).
v(X) :- e(X, _).
s(X) :- e(_, X).