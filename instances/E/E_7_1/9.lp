%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:3
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G).

group(a).
group(b).
1 { set(X, G) : group(G) } 1 :- vertex(X).
 :- edge(X, Y), not edge(X, G), -set(Y, G), group(G).