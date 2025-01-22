%formhe-problem:B
%formhe-timestamp:1698328954
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

set(S) :- e(S, _).
k { sel(S) : set(S) } k.
 :- sel(X), sel(S2), e(S1, I1), S2 != X, S1 != S2.