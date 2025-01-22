%formhe-problem:A
%formhe-timestamp:1667761693
%formhe-selfeval-lines:(), (1, 2)
%formhe-selfeval-fix:"v(N) :- e(_, N). v(N) :- e(N, _). assign(X, I) :- color(X, I).", "v(N) :- e(_, N). v(N) :- e(N, _). 1 {assign(X,I) : c(I)} 1 :- v(X). :- assign(X,I), assign(Y,I), e(X,Y), c(I)."

% Mooshak-exercises in lab13 from ALC
% Problem A

% % Default
% #const n = 3.
% colours(1..k)

% Generate
% { assign(X,1..coulours) } = 1 :- node(X).
% Test
% :- e(X,Y), assign(X,C), assign(Y,C).

% % Nodes
% node(1..6).
% Edges
% e(1,(2;3;4)).  e(2,(4;5;6)).  e(3,(1;4;5)).
% e(4,(1;2)).    e(5,(3;4;6)).  e(6,(2;3;5)).

% % Display
% #show color/2.


c(1..k).
1 {color(X,I) : c(I)} 1 :- v(X).
:- color(X,I), color(Y,I), e(X,Y), c(I).