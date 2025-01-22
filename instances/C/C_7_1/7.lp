%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- not cover(N), e(N, _).

set(X) :- e(_, X).
 :- not cover(1), e(N, _).
cover(N) :- sel(X), e(N, X).
k { sel(X) : set(X) } k.