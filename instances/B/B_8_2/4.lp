%formhe-problem:B
%formhe-timestamp:1667564976
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2.

k { sel(SET) : e(SET, _) } k.
 :- sel(SET1), sel(SET2), 1 == V1, e(V2, _), SET1 == V2, SET1 != V1.
 :- V2 == 1.