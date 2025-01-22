%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:allElements(E) :- e(E, SET), set(SET). k { sel(S) : set(S) } k. set(SET) :- e(_, SET).

 :- allElements(E), not elementsInSolution(E).
elementsInSolution(E) :- e(E, SET), sel(SET).
elementsInSolution(E) :- e(E, SET), set(SET).
set(1) :- e(_, SET).