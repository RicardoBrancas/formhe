%formhe-problem:B
%formhe-timestamp:1667564976
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2.

k { sel(SET) : e(SET, _) } k.
SET != SET1 :- sel(SET1), sel(SET2), SET1 == I1, e(V1, V2), e(V1, SET1), V2 != SET2.