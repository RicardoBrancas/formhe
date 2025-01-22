%formhe-problem:C
%formhe-timestamp:1667930682
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix: :- sel(X), sel(Y), not e(E, X), not e(E, Y), X != Y, set(X), set(Y), element(E). set(X) :- e(E, X).

element(E) :- e(E, X).
k { sel(X) : set(X) } k.
 :- sel(X), sel(Y), not e(Y, X), -e(E, Y), k != Y, set(X), set(Y), element(E).
set(X) :- e(I1, X).