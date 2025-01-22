%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:2
%formhe-selfeval-fix:covered(E) :- e(E, S), sel(S).

 :- not covered(E), e(E, S).
k { sel(S) : e(E, S) } k.
sel(E) :- e(E, E), sel(I0).