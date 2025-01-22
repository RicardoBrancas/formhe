%formhe-problem:B
%formhe-timestamp:1667732830
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- sel(X), sel(Y), e(X, N), e(Y, N), X != Y. k { sel(X) : set(X) } k.

set(X) :- e(X, _).
 :- sel(X), sel(Y), e(X, I1), e(Y, N), X != Y.
k { sel(X) : set(X) } k :- set(0; 1; (1..k)).