%formhe-problem:C
%formhe-timestamp:1667930682
%formhe-selfeval-lines:0 2 3
%formhe-selfeval-fix:element(E) :- e(E, X).  :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E). k { sel(X) : set(X) } k.

element(X) :- e(E, X).
set(X) :- e(E, X).
 :- sel(X), sel(Y), not e(E, X), not sel(E), e(X, Y), set(_), sel(Y), element(E).
k { sel(X) : sel(X) } k.