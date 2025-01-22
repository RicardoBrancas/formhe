%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- not covered(E), e(E, S). covered(E) :- e(E, S), sel(S).

 :- -covered(E), e(E, S).
k { sel(S) : e(E, S) } k.
covered(E) :- e(E, _), sel(S).