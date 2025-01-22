%formhe-problem:C
%formhe-timestamp:1667785520
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.

k { sel(SET) : e(_, SET) } k.
 :- sel(SET1), sel(SET2), 0 == SET1, SET2 != SET2, SET2 != 0, e((0..1), 1; 1).