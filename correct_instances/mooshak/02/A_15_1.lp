%formhe-problem:A
%formhe-timestamp:1699275400

v(V):-e(V,_).
v(V):-e(_,V).
1{assign(V, 1..k)}1:-v(V).
:- e(U,V), assign(U,C), assign(V,C).