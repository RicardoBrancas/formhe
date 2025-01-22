%formhe-problem:B
%formhe-timestamp:1698330801

k { sel(S) : e(S, _)} k.

:- sel(S1), sel(S2), e(S1, E1), e(S2, E1), S1 != S2.

#show sel/1.