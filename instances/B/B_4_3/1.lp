%formhe-problem:B
%formhe-timestamp:1667774702
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2.

 :- sel(SET1), sel(V1), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != 0.
k { sel(SET) : e(SET, _) } k.