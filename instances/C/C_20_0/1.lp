%formhe-problem:C
%formhe-timestamp:1698332121
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:set(S) :- e(X, S).  :- sel(S), not set(S).

element(X) :- e(X, S).
 :- e(X, S).
 :- sel(S), -set(S).
1 { sel(S) : e(X, S) }  :- element(X).
k { sel(S) : set(S) } k.