%formhe-problem:C
%formhe-timestamp:1698335456

covered(E) :- e(E, S), sel(S).

k { sel(S) : e(E, S) } k.

:- not covered(E), e(E, S).

#show sel/1.