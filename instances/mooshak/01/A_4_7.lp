%formhe-problem:A
%formhe-timestamp:1667764287
%formhe-selfeval-lines:(0, 3), (0,1,2,3)
%formhe-selfeval-fix:"c(1..k). v(N) :- e(_, N). v(N) :- e(N, _). assign(X, I) :- color(X, I).","c(1..k). v(N) :- e(_, N). v(N) :- e(N, _). 1 {assign(X,I) : c(I)} 1 :- v(X). :- assign(X,I), assign(Y,I), e(X,Y), c(I)."

% % Mooshak-exercises in lab13 from ALC
% % Problem A

% % facts
% col(1..k).

% % generate
% { assign(X,C) : col(C) } == 1 :- node(X).

% % test
% :- edge(X,Y), assign(X,C), assign(Y,C).

c(1..n).
1 {color(X,I) : c(I)} 1 :- v(X).
:- color(X,I), color(Y,I), e(X,Y), c(I).

v(1..n). % 1,...,100 are vertices
% e(1,55). % there is an edge from 1 to 55