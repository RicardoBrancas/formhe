%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y.

set(X) :- e(X, _).
 :- sel(I0), set(Y), e(X, I1), e(Y, N), e(X, Y).
k { sel(X) : set(X) } k.