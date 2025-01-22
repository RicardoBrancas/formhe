%formhe-problem:C
%formhe-timestamp:1667579913
%formhe-selfeval-lines:0
%formhe-selfeval-fix:not sel(SET1) :- sel(SET2), contains(SET1, SET2, V), SET1 != SET2.

not sel(SET1) :- sel(SET2), contains(SET1, SET2, (0..0)), contains(0, SET2, 1; 0; (k..0)).
k { sel(SET) : e(_, SET) } k.
contains(SET1, SET2, V) :- e(V, SET1), e(V, SET2), SET1 != SET2.