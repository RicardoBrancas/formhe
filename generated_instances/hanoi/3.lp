%formhe-groundtruth:../../examples/hanoi.lp
#program base.
#show move/3.
peg(a;b;c).
disk((1..4)).
init_on((1..4),a).
goal_on((1..4),c).
moves(15).
1 = { move(D,P,T): disk(D), peg(P) } :- moves(M); T = (1..M).
move_2(D,T) :- move(D,_,T).
on(D,P,0) :- init_on(D,P).
on(D,P,T) :- move(D,P,T).
on(D,P,(T+1)) :- on(D,P,T); not move_2(D,(T+1)); not moves(T).
blocked((D-1),P,(T+1)) :- on(D,P,T); not moves(T).
blocked((D-1),P,T) :- blocked(D,P,T); disk(D).
% selected out statement:
%#false :- 1 != { on(D,P,T) }; disk(D); moves(M); T = (1..M).
#false :- move(D,P,T); blocked((D-1),P,T).
#false :- move_2(D,T); on(D,P,(T-1)); blocked(D,P,T).
#false :- goal_on(D,P); not on(D,P,M); moves(M).