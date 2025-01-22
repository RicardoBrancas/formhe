%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:cover(N) :- sel(X), e(N, X). k { sel(X) : set(X) } k.

set(X) :- e(_, X).
 :- not cover(N), e(N, _).
X == X :- sel(X), e(N, N).
cover(k).