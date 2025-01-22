%formhe-problem:B
%formhe-timestamp:1698330800
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, E1), e(S2, E1), S1 != S2.

k { sel(S) : e(S, _) } k.
 :- sel(S1), sel(I1), e(S1, S2), e(_, _), k == S2, sel(_).