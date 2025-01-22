%formhe-problem:C
%formhe-timestamp:1668360092
%formhe-selfeval-lines:0
%formhe-selfeval-fix:k { sel(A) : e(Y, A) } k.

 { sel(A) : e(_, A) } 1.
-sel(B) :- e(Y, A), e(Y, B), sel(A), A != B.