%formhe-problem:B
%formhe-timestamp:1667674665

disjunct(X,Y) :- e(X,Z), e(X,C), e(Y,V), e(Y,W), Z!=V, C!=W, Z!=W, Z!=C, V!=W, C!=V, X!=Y.
{sel(X)} :- disjunct(X,Y).
total(S) :- S = #count{ X : e(X,Y), sel(X)}.
:- total(X), X != k.