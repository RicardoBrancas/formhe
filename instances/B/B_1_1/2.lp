%formhe-problem:B
%formhe-timestamp:1668364658
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(A) : e(A, Y) } k.

-sel(A) :- e(A, Y), e(B, Y), sel(B), A != B.
k { sel(Y) : e(A, Y) } k.