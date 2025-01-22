%formhe-problem:B
%formhe-timestamp:1668364659

-sel(A) :- e(A,Y), e(B,Y), sel(B), A!=B.
k {sel(A) : e(A,Y)} k.