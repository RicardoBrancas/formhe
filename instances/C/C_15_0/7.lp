%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:1 4
%formhe-selfeval-fix:elementsInSolution(E) :- e(E, SET), sel(SET). set(SET) :- e(_, SET).

 :- allElements(E), not elementsInSolution(E).
 :- e(E, SET), sel(SET).
allElements(E) :- e(E, SET), set(SET).
k { sel(S) : set(S) } k.
set(SET) :- e(0, SET).