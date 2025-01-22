%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

 :- sel(X), sel(Y), e(I1, N), e(Y, N), X == Y, e(1, k).
k { sel(X) : set(X) } k.
set(X) :- e(X, _).