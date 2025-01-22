%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:cover(N) :- sel(X), e(N, X). k { sel(X) : set(X) } k.  :- not cover(N), e(N, _).

set(X) :- e(_, X).
set(N) :- sel(X), cover(N).
k { sel(X) : set(X) } 0 :- e((k..1), (1..0)).
 :- -cover(N), e(N, _).