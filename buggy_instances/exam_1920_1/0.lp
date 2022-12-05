%formhe-groundtruth:../../examples/exam_1920_1.lp

#const k = 2.

s(X) :- e(X, E).

k { sel(X) : s(X) } k.

inter(X, Y) :- e(X, E), e(Y, E), X != Y.

:- inter(X, Y), s(X), s(Y).