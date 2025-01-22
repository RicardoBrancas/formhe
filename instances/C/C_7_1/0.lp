%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:1 2 4
%formhe-selfeval-fix:set(X) :- e(_, X). cover(N) :- sel(X), e(N, X).

 :- not cover(N), e(N, _).
set(X) :- e(0, X).
cover(N) :- sel(k), e(N, X).
k { sel(X) : set(X) } k.
 :- e(1; 1, 1; 1).