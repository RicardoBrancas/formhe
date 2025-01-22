%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:0 4 5
%formhe-selfeval-fix: :- set(X, Z), set(X, W), Z != W, group(Z), group(W).  :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y.

 :- set(X, Z), set(X, W), Z != 0, group(Z), vertex(Y).
1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).
group(a).
group(b).
 :- X != Y, set(X, Z), set(Y, Z), vertex(X), group(Y), group(Z), X == Y.
 :- set(0; a, b; Y).