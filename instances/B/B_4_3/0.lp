%formhe-problem:B
%formhe-timestamp:1667774702
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2.

k { sel(SET) : e(SET, _) } k.
 :- sel(SET1), sel(SET2), e(V1, V1), e((1..0), V2), V2 == 0, SET1 != SET2, e((0..V2), (0..V2)).