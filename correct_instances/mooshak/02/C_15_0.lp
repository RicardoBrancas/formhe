%formhe-problem:C
%formhe-timestamp:1699275400

set(SET):- e(_, SET).
k {sel(S): set(S)} k.
elementsInSolution(E):-e(E,SET),sel(SET).
allElements(E):-e(E,SET),set(SET).
:-allElements(E), not elementsInSolution(E).