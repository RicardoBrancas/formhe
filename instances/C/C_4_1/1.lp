%formhe-problem:C
%formhe-timestamp:1667785520
%formhe-selfeval-lines:0 2
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.

 :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, I0), SET1 != SET2, 0 == 0.
k { sel(SET) : e(_, SET) } k.
 :- e((k..0), (0..1)).