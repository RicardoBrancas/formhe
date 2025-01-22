%formhe-problem:B
%formhe-timestamp:1667674664
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix:disjunct(X, Y) :- e(X, Z), e(X, C), e(Y, V), e(Y, W), Z != V, C != W, Z != W, Z != C, V != W, C != V, X != Y.  { sel(X) }  :- disjunct(X, Y). total(S) :- S = #count { X: e(X,Y), sel(X) }.

disjunct(X, Y) :- e(C, Z), e(X, C), e(Y, V), e(Y, W), Z != 0, C != W, Z != W, Z != C, V != W, C != V, X != Y.
 { sel(X) } 0 :- disjunct(X, Y).
sel(k) :- total(Count).
 :- total(X), X != k.