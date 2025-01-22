%formhe-problem:C
%formhe-timestamp:1667579913
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(SET) : e(_, SET) } k. contains(SET1, SET2, V) :- e(V, SET1), e(V, SET2), SET1 != SET2.

not sel(SET1) :- sel(SET2), contains(SET1, SET2, V), SET1 != SET2.
 :- e(V, SET1), V == SET, contains(SET, SET2, 0).