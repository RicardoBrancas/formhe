%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- not covered(E), e(E, S).

k { sel(S) : e(E, S) } k.
 :- not covered(E), e(E, 1).
covered(E) :- e(E, S), sel(S).