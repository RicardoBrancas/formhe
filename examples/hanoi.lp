%%%%%%%%%%
% Instance
peg(a;b;c).
disk(1..4).
init_on(1..4,a).
goal_on(1..4,c).
moves(15).
%%%%%%%%%%
% Generate
% at each point T in time (other than 0), exactly one
% move of a disk D to some peg P must be executed.
{ move(D,P,T) : disk(D), peg(P) } = 1 :- moves(M), T =
1..M.
%%%%%%%%
% Define
% projection
move_2(D,T) :- move(D,_,T).
% capture state of towers
% start
on(D,P,0) :- init_on(D,P).
% move
on(D,P,T) :- move(D,P,T).
% inertia
on(D,P,T+1) :- on(D,P,T), not move_2(D,T+1), not moves(T).
% a smaller disk, with a number greater than D-1, is located on a peg P
blocked(D-1,P,T+1) :- on(D,P,T), not moves(T).
% propagate to larger disks
blocked(D-1,P,T) :- blocked(D,P,T), disk(D).
%%%%%%
% Test
% a disk D must not be moved to a peg P if D-1 is blocked at time point T
:- move(D,P,T), blocked(D-1,P,T).
% a disk D cannot be moved at time point T if it is blocked by
% some smaller disk on the same peg P
:- move_2(D,T), on(D,P,T-1), blocked(D,P,T).
% the goal situation, given in an instance, must be achieved at
% maximum time point M
:- goal_on(D,P), not on(D,P,M), moves(M).
% for every disk D and time point T, there is exactly one peg P
% such that on(D,P,T) holds
:- { on(D,P,T) } != 1, disk(D), moves(M), T = 1..M.
% note that this is already implied but adding it improves performance
%%%%%%%%%
% Display
#show move/3.