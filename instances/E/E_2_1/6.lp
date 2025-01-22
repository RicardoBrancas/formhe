%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:3 4
%formhe-selfeval-fix: :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y.  :- set(X, Z), set(X, W), Z != W, group(Z), group(W).

group(a).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).
group(b).
 :- set(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), vertex(Z), X != Y.
 :- set(X, Z), set(X, W), Z != W, vertex(Z), set(W, a).