%formhe-problem:E
%formhe-timestamp:1667941045
%formhe-selfeval-lines:0 1 4
%formhe-selfeval-fix: :- set(X, Z), set(X, W), Z != W, group(Z), group(W).  :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y. 1 { set(X, Z) : vertex(X), group(Z) } 1 :- vertex(X).

 :- set(X, Z), set(X, W), b != W, group(Z), group(W).
 :- edge(X, Y), set(X, Z), set(Y, Z), vertex(X), vertex(Y), group(Z), X != Y, set(1; 0, 0; b).
group(b).
group(a).
1 { set(X, Z) : vertex(X), vertex(Z) } 1 :- vertex(X).