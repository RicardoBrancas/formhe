%formhe-problem:B
%formhe-timestamp:1698328954
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:k { sel(S) : set(S) } k.  :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

set(S) :- e(S, _).
k { sel(S) : set(S) } 1.
 :- sel(X), sel(S2), e(S1, S2), e(S2, S1), S1 != S2.