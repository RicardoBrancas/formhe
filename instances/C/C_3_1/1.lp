%formhe-problem:C
%formhe-timestamp:1667675682
%formhe-selfeval-lines:0
%formhe-selfeval-fix:total(S) :- S = #count { X: e(Y,X), sel(X) }.

total(S) :- cover(Count), cover(S).
 :- e(X, Y), not cover(X).
cover(X) :- e(X, Y), sel(Y).
 :- total(S), S != k.
 { sel(X) }  :- e(Y, X).