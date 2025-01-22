%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:1 3 4
%formhe-selfeval-fix: { sel(V) : v(V) } k. v(X) :- e(_, X). conected(V1, V2) :- e(V2, V1).  :- conected(V1, V2), not sel(V1), not sel(V2).

conected(V1, V2) :- e(V1, V2).
v(X) :- e(_, X), 0 == X.
v(X) :- e(X, _).
sel(V1) :- e(V2, V1), e(0, _).
 :- conected(V1, V2), not v(V1), not sel(V2), conected(0; V2, V2; X).