%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- conected(V1, V2), not sel(V1), not sel(V2).

conected(V1, V2) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), not sel(V2), conected((1..k), (1..k)).
v(X) :- e(_, X).
v(X) :- e(X, _).
 { sel(V) : v(V) } k.
conected(V1, V2) :- e(V1, V2).