%formhe-problem:B
%formhe-timestamp:1698329138
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(S) : set(S) } k.  :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

set(S) :- e(S, _).
 :- sel(S1), sel(S1), e(S1, S2), S2 == X, S == S2.