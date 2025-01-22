%formhe-problem:A
%formhe-timestamp:1698332122

color(1..k).
vertex(U) :- e(U,V).
vertex(V) :- e(U,V).
1{ assign(V,C) : color(C) }1 :- vertex(V).
:- e(V1,V2), assign(V1,C), assign(V2,C).
#show assign/2.