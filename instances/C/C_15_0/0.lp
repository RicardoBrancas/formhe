%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- allElements(E), not elementsInSolution(E).

set(SET) :- e(_, SET).
 :- allElements(E), not allElements(E).
elementsInSolution(E) :- e(E, SET), sel(SET).
allElements(E) :- e(E, SET), set(SET).
k { sel(S) : set(S) } k.