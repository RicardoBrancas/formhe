%formhe-problem:C
%formhe-timestamp:1698332121
%formhe-selfeval-lines:
%formhe-selfeval-fix:k { sel(S) : set(S) } k.

element(X) :- e(X, S).
set(S) :- e(X, S).
1 { sel(S) : e(X, S) }  :- element(X).
 :- sel(S), not set(S).