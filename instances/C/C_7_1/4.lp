%formhe-problem:C
%formhe-timestamp:1667734194
%formhe-selfeval-lines:3
%formhe-selfeval-fix:k { sel(X) : set(X) } k.

 :- not cover(N), e(N, _).
set(X) :- e(_, X).
cover(N) :- sel(X), e(N, X).
k { e(X, (0..1)) : set(X) } k.