%formhe-problem:B
%formhe-timestamp:1698328954
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

set(S) :- e(S, _).
 :- sel(S), set(S2), e(S1, X), e(I1, X), S != I1.
k { sel(S) : set(S) } k.