%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:0
%formhe-selfeval-fix:set(SET) :- e(_, SET).

set(SET) :- e(1, SET).
allElements(E) :- e(E, SET), set(SET).
k { sel(S) : set(S) } k.
elementsInSolution(E) :- e(E, SET), sel(SET).
 :- allElements(E), not elementsInSolution(E).