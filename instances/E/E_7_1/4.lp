%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G). 1 { set(X, G) : group(G) } 1 :- vertex(X).

 :- edge(X, Y), not set(X, G), not set(G, G), group(G).
1 { set(X, G) : group(G) } b :- vertex(X).
group(a).
group(b).