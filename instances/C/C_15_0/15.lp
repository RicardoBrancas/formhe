%formhe-groundtruth:../../../mooshak/instances/1819_2.lp
%formhe-domain-predicates:sel/1
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSg0LCBjKS4KZSg1LCBjKS4=
%formhe-instance-base64:I2NvbnN0IGsgPSAyLgoKZSgxLCBhKS4KZSgyLCBhKS4KZSgzLCBhKS4KZSgyLCBiKS4KZSg0LCBiKS4KZSgzLCBjKS4KZSg0LCBjKS4KZSg0LCBkKS4KZSg1LCBkKS4=
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix:allElements(E) :- e(E, SET), set(SET).  :- allElements(E), not elementsInSolution(E).

set(SET) :- e(_, SET).
k { sel(S) : set(S) } k.
elementsInSolution(E) :- e(E, SET), sel(SET).
allElements(E) :- e(E, SET), allElements(SET).
0 != I0 :- allElements(E), -elementsInSolution(E).