%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- set(X, Z), set(X, W), Z != W, group(Z), group(W).  :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y. group(a).

 :- set(I1, Z), X == W, Z != W, group(Z), group(W).
 :- edge(X, Y), set(X, W), set(Y, Z), vertex(X), vertex(Y), group(Z), X == Y.
group(b) :- 0 == 1.
group(b).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).