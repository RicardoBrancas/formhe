%formhe-problem:C
%formhe-timestamp:1668357843

sel(A) :- e(Y,A), e(Y,B), not sel(B).
sel(A) :- e(Y,A), not e(Y,B).