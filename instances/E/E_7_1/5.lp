%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:0 1 2
%formhe-selfeval-fix: :- edge(X, Y), not set(X, G), not set(Y, G), group(G). 1 { set(X, G) : group(G) } 1 :- vertex(X). group(a).

 :- edge(X, Y), not set(X, G), not set(Y, b), group(G).
1 { set(X, a) : group(a) } b :- vertex(X).
vertex(a).
group(b).