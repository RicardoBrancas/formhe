%formhe-problem:C
%formhe-timestamp:1698332764
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- e(E, _),  { sel(S) : e(E, S) } 0.

k { sel(S) : e(_, S) } k.
 :- e(E, _), not sel(_).