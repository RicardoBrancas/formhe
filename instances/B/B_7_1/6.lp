%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:set(X) :- e(X, _).  :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

set(0) :- e(N, _).
k { sel(X) : set(X) } k.
 :- sel(X), sel(X), e(X, N), e(N, N), X != N.