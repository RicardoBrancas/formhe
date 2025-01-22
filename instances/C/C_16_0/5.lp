%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:k { sel(S) : e(E, S) } k.  :- not covered(E), e(E, S).

covered(E) :- e(E, S), sel(S).
k { covered(S) : e(E, S) } 0.
 :- -sel(_), e(E, E).