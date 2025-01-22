%formhe-problem:D
%formhe-timestamp:1667586028
%formhe-selfeval-lines:(), (0)
%formhe-selfeval-fix:"v(I0) :- e(I0, _). v(I0) :- e(_, I0).","{ sel(V;U): e(V,U) } k."

{ sel(V): v(V) } k.

conected(V1, V2):- e(V1, V2).
conected(V1, V2):- e(V2, V1).



:- conected(V1, V2), not sel(V1), not sel(V2).