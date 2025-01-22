%formhe-problem:D
%formhe-timestamp:1668364992
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix::- not sel(X), not sel(Y), e(X,Y). 0 {sel(A;B) : e(A,B)} k.

sel(A) :- e(A,B), e(B,A).
k {sel(A) : e(A,B)} k.