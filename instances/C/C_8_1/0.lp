%formhe-problem:C
%formhe-timestamp:1667579913
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:contains(SET1, SET2, V) :- e(V, SET1), e(V, SET2), SET1 != SET2. k { sel(SET) : e(_, SET) } k.

not sel(SET1) :- sel(SET2), contains(SET1, SET2, V), SET1 != SET2.
 :- e(V, SET1), e(SET1, SET2), SET1 == SET, sel(I0; k).
k { sel(SET) : e(0, SET) } .