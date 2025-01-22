%formhe-problem:B
%formhe-timestamp:1698329138
%formhe-selfeval-lines:2
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

set(S) :- e(S, _).
k { sel(S) : set(S) } k.
 :- sel(S1), sel(S2), e(S, 0), S2 == X, S1 != S2.