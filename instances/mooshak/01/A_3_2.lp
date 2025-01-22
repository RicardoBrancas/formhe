%formhe-problem:A
%formhe-timestamp:1667498807

1{assign(X:Y,1..k)}1 :- e(X,Y).
:- e(X,Y), assign(X,C), assign(Y,C).