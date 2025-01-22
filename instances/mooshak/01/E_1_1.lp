%formhe-problem:E
%formhe-timestamp:1668374341
%formhe-selfeval-lines:2 3
%formhe-selfeval-fix::- edge(A,B), set(A, a), set(B, a). :- edge(A,B), set(A, b), set(B, b).

set(A,a) | set(A,b) :- edge(A,_).
set(A,a) | set(A,b) :- edge(_,A).


:- edge(A,B), set(A, a), set(B, b).
:- edge(A,B), set(A, b), set(B, a).