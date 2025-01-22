%formhe-problem:E
%formhe-timestamp:1667812517

group(a). group(b).

1{set(X, G) : group(G)}1:-vertex(X).

:- edge(X,Y), not set(X, G), not set(Y,G), group(G).