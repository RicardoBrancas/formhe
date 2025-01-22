%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:1 2 4
%formhe-selfeval-fix:allElements(E) :- e(E, SET), set(SET). elementsInSolution(E) :- e(E, SET), sel(SET).  :- allElements(E), not elementsInSolution(E).

k { sel(S) : set(S) } k.
allElements(E) :- e(E, 0), set(_).
elementsInSolution(E) :- E == SET, set(SET).
set(SET) :- e(_, SET).
 :- allElements(E), -elementsInSolution(E).