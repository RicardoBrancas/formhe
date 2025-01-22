%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:1
%formhe-selfeval-fix:cover(N) :- sel(X), e(N, X).

k { sel(X) : set(X) } k.
cover(X) :- sel(X), e(N, X).
set(X) :- e(_, X).
 :- not cover(N), e(N, _).