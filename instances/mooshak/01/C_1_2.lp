%formhe-problem:C
%formhe-timestamp:1668357653
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:k {sel(A) : e(Y,A)} k. cov(X) :- e(X, S), sel(S). :- not cov(X), e(X, _).

sel(A) :- e(Y,A), -e(Y,B).
sel(A) :- e(Y,A), e(Y,B), not sel(B).