%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:0
%formhe-selfeval-fix:k { sel(X) : set(X) } k.

k { sel(X) : set(X) }  :- e(1; 1, 1; 1).
cover(N) :- sel(X), e(N, X).
set(X) :- e(_, X).
 :- not cover(N), e(N, _).