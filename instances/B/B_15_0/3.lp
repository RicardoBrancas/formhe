%formhe-problem:B
%formhe-timestamp:1699275399
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, E), e(S2, E), S1 != S2.

set(S) :- e(S, _).
0 == 1 :- sel(S1), sel(S2), S1 == I1, e(S2, S2), S1 != I1.
k { sel(S) : set(S) } k.