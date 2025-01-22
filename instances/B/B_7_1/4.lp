%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:k { sel(X) : set(X) } k.  :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

 { sel(X) : set(X) } k.
set(X) :- e(X, _).
 :- sel(_), sel(Y), I1 == N, e(Y, N), 0 != Y.