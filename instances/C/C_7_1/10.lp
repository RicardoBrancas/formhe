%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- not cover(N), e(N, _). cover(N) :- sel(X), e(N, X).

set(X) :- e(_, X).
k { sel(X) : set(X) } k.
set((k..N)) :- -cover(N), e(N, _).
cover(N) :- set(I1), N == 1, e((k..1), (1..0)).