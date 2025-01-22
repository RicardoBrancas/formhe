%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:k { sel(X) : set(X) } k.  :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

k { sel(X) : set(X) } .
 :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y, X == 0.
set(X) :- e(X, _).