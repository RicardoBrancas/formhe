%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y.

 :- set(X, Z), set(X, W), Z != W, group(Z), group(W).
 :- edge(X, Y), set(X, Z), set(Y, _), vertex(X), vertex(Y), group(b), X != Y.
group(a).
group(b).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).