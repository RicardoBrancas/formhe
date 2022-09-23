%formhe-groundtruth:../../examples/exam_1920_2.lp
%formhe-mcs-query:assign(a, 1). assign(b, 2).

#const k = 3.

formhe_definition_begin.

e(a, b).
e(a, c).
e(a, d).
e(b, c).
e(c, d).

formhe_definition_end.

color(1..k).

node(N) :- e(_, N).
node(N) :- e(N, _).

1 { assign(N, C) : color(C) } 1 :- node(N).

:- assign(N, C), assign(M, C).