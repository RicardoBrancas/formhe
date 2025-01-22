%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- not cover(N), e(N, _).

 :- not e(N, (1..0)), e(N, _).
set(X) :- e(_, X).
cover(N) :- sel(X), e(N, X).
k { sel(X) : set(X) } k.