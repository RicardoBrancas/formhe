%formhe-problem:C
%formhe-timestamp:1667785520
%formhe-selfeval-lines:0
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.

 :- sel(ELEMENT), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.
k { sel(SET) : e(_, SET) } k.