%formhe-problem:E
%formhe-timestamp:1668374443

set(A,a) | set(A,b) :- edge(A,_).
set(A,a) | set(A,b) :- edge(_,A).


:- edge(A,B), set(A, a), set(B, a).
:- edge(A,B), set(A, b), set(B, b).