%formhe-problem:D
%formhe-timestamp:1667586067
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:v(X) :- e(X, _).  { sel(V) : v(V) } k. conected(V1, V2) :- e(V1, V2).

v(X) :- e(_, X).
v(X) :- conected(X, _), e(0; 0, 0; k).
 { sel(V) : v(V) } .
e(V2, 0; 0) :- e(V1, V2).
conected(V1, V2) :- e(V2, V1).
 :- conected(V1, V2), not sel(V1), not sel(V2).