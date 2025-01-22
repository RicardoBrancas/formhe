%formhe-problem:C
%formhe-timestamp:1667785520
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix: :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET2), SET1 != SET2.

k { sel(SET) : e(_, SET) } k.
 :- sel(SET1), sel(SET2), e(ELEMENT, SET1), e(ELEMENT, SET), SET1 != SET2.
 :- e(0; SET1, ELEMENT; ELEMENT).