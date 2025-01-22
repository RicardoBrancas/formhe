%formhe-problem:C
%formhe-timestamp:1667785520
%formhe-selfeval-lines:1
%formhe-selfeval-fix:k { sel(SET) : e(_, SET) } k.

 :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.
k { e(SET, (0..1)) : e(_, SET) } k.