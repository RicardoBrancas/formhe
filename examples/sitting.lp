formhe_definition_begin.

person(a).
person(b).
person(c).
person(d).
likes(a,b).
likes(c,d).
dislikes(a,c).
tbl(1,3).
tbl(2,2).

formhe_definition_end.

1 { seat(P, T) : tbl(T, _) } 1 :- person(P).

:- tbl(T,C), C + 1 { seat(P, T) : person(P) }.
:- person(P1), person(P2), tbl(T1, _), tbl(T2,_), likes(P1, P2), seat(P1,T1), seat(P2,T2), T1 != T2.
:- person(P1), person(P2), tbl(T, _), dislikes(P1, P2), seat(P1,T), seat(P2,T).