%formhe-problem:B
%formhe-timestamp:1698330800
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, E1), e(S2, E1), S1 != S2.

k { sel(S) : e(S, _) } k.
0 != 1 :- sel(S1), sel(E1), e(S1, 0), e(S2, _), S1 == S2.