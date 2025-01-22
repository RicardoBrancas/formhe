%formhe-problem:C
%formhe-timestamp:1667675682
%formhe-selfeval-lines:1 4
%formhe-selfeval-fix:total(S) :- S = #count { X: e(Y,X), sel(X) }.  { sel(X) }  :- e(Y, X).

 :- e(X, Y), not cover(X).
total(S) :- sel(S).
cover(X) :- e(X, Y), sel(Y).
 :- total(S), S != k.
 { sel(X) } Y :- e(Y, X).