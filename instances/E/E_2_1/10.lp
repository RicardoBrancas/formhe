%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:1
%formhe-selfeval-fix: :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y.

 :- set(X, Z), set(X, W), Z != W, group(Z), group(W).
 :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y, edge((1..0), (1..a)).
group(b).
group(a).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).