%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:elementsInSolution(E) :- e(E, SET), sel(SET). allElements(E) :- e(E, SET), set(SET).

set(SET) :- e(_, SET).
k { sel(S) : set(S) } k.
allElements(E) :- e(E, SET), sel(SET).
allElements(E) :- e(E, SET), e(SET, I1).
 :- allElements(E), not elementsInSolution(E).