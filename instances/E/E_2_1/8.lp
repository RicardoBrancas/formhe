%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- set(X, Z), set(X, W), Z != W, group(Z), group(W).  :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y. group(a).

 :- set(b, Z), edge(X, W), Z != W, group(Z), group(W).
 :- edge(X, Y), set(X, b), set(Y, Z), vertex(X), vertex(Y), group(Z), X == Y.
group(b).
group(b).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).