%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- conected(V1, V2), not sel(V1), not sel(V2).

v(X) :- e(_, X).
 { sel(V) : v(V) } k.
conected(V1, V2) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), -sel(V2).
v(X) :- e(X, _).
conected(V1, V2) :- e(V1, V2).