%formhe-problem:B
%formhe-timestamp:1698328954
%formhe-selfeval-lines:0
%formhe-selfeval-fix:set(S) :- e(S, _). k { sel(S) : set(S) } k.

set(S) :- e(S, I1).
 :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.