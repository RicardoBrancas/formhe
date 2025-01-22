%formhe-problem:A
%formhe-timestamp:1698330475

color(1..k).

vertex(X) :- e(X, Y).
vertex(Y) :- e(X, Y).

1 { assign(V, C) : color(C)} 1 :-  vertex(V).

:- e(X, Y), assign(X, C), assign(Y, C).

#show assign/2.