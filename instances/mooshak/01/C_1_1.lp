%formhe-problem:C
%formhe-timestamp:1668356664
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:k {sel(A) : e(Y,A)} k. cov(X) :- e(X, S), sel(S). :- not cov(X), e(X, _).

sel(A) :- e(Y,A).
sel(B) :- e(Y,B), -e(Y,A).