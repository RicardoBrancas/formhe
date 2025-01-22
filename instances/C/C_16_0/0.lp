%formhe-problem:C
%formhe-timestamp:1698335455
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix:k { sel(S) : e(E, S) } k.  :- not covered(E), e(E, S).

k { sel(S) : e(S, S) } k.
covered(E) :- e(E, S), sel(S).
e((1..0), (0..1)) :- not covered(E), e(E, S).