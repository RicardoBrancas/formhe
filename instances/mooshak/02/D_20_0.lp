%formhe-problem:D
%formhe-timestamp:1698332122
%formhe-selfeval-lines:3
%formhe-selfeval-fix:1{ sel(V) : vertex(V) }k.

vertex(U) :- e(U,V).
vertex(V) :- e(U,V).
:- e(U,V), not sel(U), not sel(V).
k{ sel(V) : vertex(V) }k.
#show sel/1.