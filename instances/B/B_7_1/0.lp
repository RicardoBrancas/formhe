%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

k { sel(X) : set(X) } k.
 :- sel(X), sel(Y), e(X, N), I1 == N, X != Y.
set(X) :- e(X, _).