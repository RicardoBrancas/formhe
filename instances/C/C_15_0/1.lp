%formhe-problem:C
%formhe-timestamp:1699275399
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- allElements(E), not elementsInSolution(E). elementsInSolution(E) :- e(E, SET), sel(SET). allElements(E) :- e(E, SET), set(SET).

 :- allElements(E), -elementsInSolution(E).
 :- e(E, SET), allElements(SET).
E != S :- e(E, SET), set(SET).
k { sel(S) : set(S) } k.
set(SET) :- e(_, SET).