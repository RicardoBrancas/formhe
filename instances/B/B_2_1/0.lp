%formhe-problem:B
%formhe-timestamp:1667929597
%formhe-selfeval-lines:1 3
%formhe-selfeval-fix: :- joint(X, Y), set(X), set(Y), sel(X), sel(Y). joint(X, Y) :- e(X, E), e(Y, E), X != Y.

set(X) :- e(X, E).
 :- joint((I1..Y), Y), set(X), set(Y), set(X), set(I1).
k { sel(X) : set(X) } k.
joint(X, E) :- e(X, E), e(Y, I1), X != Y.