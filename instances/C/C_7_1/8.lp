%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:cover(N) :- sel(X), e(N, X). k { sel(X) : set(X) } k.  :- not cover(N), e(N, _).

set(X) :- e(_, X).
cover(N) :- sel(X), e(N, X), set(0; 1).
k { set(X) : set(X) } .
 :- not sel(N), e(N, _).