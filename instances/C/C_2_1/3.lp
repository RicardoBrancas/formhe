%formhe-problem:C
%formhe-timestamp:1667930682
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:k { sel(X) : set(X) } k.  :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E).

element(E) :- e(E, X).
 { sel(X) : set(X) } k.
 :- sel(X), sel(Y), not sel(E), not e(E, Y), X != Y, set(X), set(Y), element(E).
set(X) :- e(E, X).