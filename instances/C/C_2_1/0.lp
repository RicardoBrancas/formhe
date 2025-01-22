%formhe-problem:C
%formhe-timestamp:1667930682
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E).

element(E) :- e(E, X).
k { sel(X) : set(X) } k.
set(X) :- e(E, X).
 :- sel(X), element(X), not e(E, X), not e(E, Y), k != Y, set(X), sel(Y), sel(E).