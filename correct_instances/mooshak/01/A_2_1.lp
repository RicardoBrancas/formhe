%formhe-problem:A
%formhe-timestamp:1667929700

{ assign(X,1..k) } = 1 :- e(X, Y).
{ assign(Y,1..k) } = 1 :- e(X, Y).
:- e(X,Y), assign(X,C), assign(Y,C).