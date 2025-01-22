%formhe-problem:E
%formhe-timestamp:1667812516
%formhe-selfeval-lines:1 2 3
%formhe-selfeval-fix:group(a). group(b).  :- edge(X, Y), not set(X, G), not set(Y, G), group(G).

1 { set(X, G) : group(G) } 1 :- vertex(X).
group(a) :- 0 == a.
group(b) :- 0 != 1.
vertex(0) :- edge(X, Y), not set(X, b), not set(Y, G), group(G).