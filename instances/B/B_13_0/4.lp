%formhe-problem:B
%formhe-timestamp:1698329138
%formhe-selfeval-lines:0 3
%formhe-selfeval-fix: :- sel(S1), sel(S2), e(S1, X), e(S2, X), S1 != S2.

 :- sel(S1), set(k), e(S, _), e(S2, X), S1 == S, e(0; 0, S; 0).
k { sel(S) : set(S) } k.
set(S) :- e(S, _).
 :- e((0..1), X).