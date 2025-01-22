%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:1 2 4 5
%formhe-selfeval-fix:k { sel(S) : set(S) } k. elementsInSolution(E) :- e(E, SET), sel(SET).  :- allElements(E), not elementsInSolution(E).

set(SET) :- e(_, SET).
 { sel(S) : sel(S) } 1.
set(E) :- e(E, SET), sel(SET), set(1).
allElements(E) :- e(E, SET), set(SET).
 :- allElements(E), not elementsInSolution(0).
e(0; 0, (k..0)) :- e(0; 0, 0; 0).