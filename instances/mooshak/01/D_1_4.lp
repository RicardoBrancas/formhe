%formhe-problem:D
%formhe-timestamp:1668371619
%formhe-selfeval-lines:0 1 2 3
%formhe-selfeval-fix::- not sel(X), not sel(Y), e(X,Y). 0 {sel(A;B) : e(B,A)} k.

sel(A) :- e(A,B).
sel(A) :- e(B,A).
k {sel(A)} k :- e(B,A).
k {sel(A)} k :- e(A,B).