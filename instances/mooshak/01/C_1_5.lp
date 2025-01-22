%formhe-problem:C
%formhe-timestamp:1668359574
%formhe-selfeval-lines:0
%formhe-selfeval-fix:cov(X) :- e(X, S), sel(S). :- not cov(X), e(X, _).

sel(A) :- e(Y,A), -e(Y,B).
k {sel(A) : e(Y,A)} k.