%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:k { sel(S) : e(E, S) } k. covered(E) :- e(E, S), sel(S).

 :- not covered(E), e(E, S).
k { sel(S) : e(_, S) } .
covered(k) :- e(0, S), sel(E).