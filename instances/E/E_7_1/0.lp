%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix:1 { set(X, G) : group(G) } 1 :- vertex(X).  :- edge(X, Y), not set(X, G), not set(Y, G), group(G).

group(a).
group(b).
1 { set(X, G) : vertex(G) } 1 :- vertex(X).
 :- set(X, b), not set(X, G), -set(Y, G), group(G).