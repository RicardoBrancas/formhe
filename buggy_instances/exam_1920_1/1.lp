%formhe-groundtruth:../../examples/exam_1920_1.lp
%formhe-mcs-query:sel(a). sel(b).

#const k = 2.

formhe_definition_begin.

e(a, 1).
e(a, 2).
e(b, 3).
e(b, 4).
e(c, 1).
e(c, 3).

formhe_definition_end.

s(X) :- e(X, E).

k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E).

:- inter(X, Y), sel(X), sel(Y).