%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:
%formhe-selfeval-fix:set(X) :- e(X, _). k { sel(X) : set(X) } k.

 :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.