%formhe-problem:C
%formhe-timestamp:1698332121
%formhe-selfeval-lines:0
%formhe-selfeval-fix:1 { sel(S) : e(X, S) }  :- element(X).

1 { sel(X) : e(X, k) }  :- element(X).
element(X) :- e(X, S).
set(S) :- e(X, S).
k { sel(S) : set(S) } k.
 :- sel(S), not set(S).