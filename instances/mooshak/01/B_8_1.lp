%formhe-problem:B
%formhe-timestamp:1667564686
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix:k { sel(I0) : e(I0, _) } k.

#const n = 1.


n {sel(SET): e(SET, _)} n .


:- sel(SET1), sel(SET2), e(SET1, V1), e(SET2, V2), V1 == V2, SET1 != SET2 .