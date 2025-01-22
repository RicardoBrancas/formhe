%formhe-problem:A
%formhe-timestamp:1667761880
%formhe-selfeval-lines:(1, 2), (0, 2)
%formhe-selfeval-fix:"node(I0) :- e(_, I0). node(I0) :- e(I0, _). 1 {assign(X,I) : color(I)} 1 :- node(X). :- assign(X,I), assign(Y,I), e(X,Y), color(I).", "node(I0) :- e(_, I0). node(I0) :- e(I0, _). c(1..k). :- assign(X,I), assign(Y,I), e(X,Y), c(I)."

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


color(1..k).
1 {assign(X,I) : c(I)} 1 :- node(X).
:- assign(X,I), assign(Y,I), node(X,Y), color(I).