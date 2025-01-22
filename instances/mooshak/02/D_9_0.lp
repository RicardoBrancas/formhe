%formhe-problem:D
%formhe-timestamp:1698336715
%formhe-selfeval-lines:1 2
%formhe-selfeval-fix:vertex(V) :- e(_,V). 1 { sel(V) : vertex(V)} k. :- e(V1,V2), not sel(V1), not sel(V2).


vertex(V) :- e(V,_).
k { sel(V) : vertex(V)} k.

:- vertex(V1), sel(V2), e(V1,V2).