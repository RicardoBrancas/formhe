%formhe-problem:C
%formhe-timestamp:1
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix:cov(X) :- v(X), e(X, S), sel(S). s(X) :- e(_, X).

 :- not cov(X), v(X).
0 { sel(X) : s(X) } k.
v(X) :- e(X, _).
cov(X) :- v(X), e(X, S), sel(I1).
s(X) :- e(_, X), e(_, _).