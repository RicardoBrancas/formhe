%formhe-problem:B
%formhe-timestamp:1667674664
%formhe-selfeval-lines:
%formhe-selfeval-fix: { sel(X) }  :- disjunct(X, Y). disjunct(X, Y) :- e(X, Z), e(X, C), e(Y, V), e(Y, W), Z != V, C != W, Z != W, Z != C, V != W, C != V, X != Y.  :- total(X), X != k.

total(S) :- S = #count { X: e(X,Y), sel(X) }.