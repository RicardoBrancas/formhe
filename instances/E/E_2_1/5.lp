%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:4
%formhe-selfeval-fix: :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y.

 :- set(X, Z), set(X, W), Z != W, group(Z), group(W).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).
group(a).
group(b).
 :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), group(Y), group(Z), X != Y.