%formhe-problem:B
%formhe-timestamp:1698330800
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, E1), e(S2, E1), S1 != S2.

 :- sel(S), e(S2, S2), e(S, S2), e(S2, E1), 0 == S1.
k { sel(S) : e(S, _) } k.