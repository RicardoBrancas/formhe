%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:0
%formhe-selfeval-fix:1 { set(X, G) : group(G) } 1 :- vertex(X).

1 { set(X, G) : group(G) }  :- vertex(X).
group(a).
group(b).
 :- edge(X, Y), not set(X, G), not set(Y, G), group(G).