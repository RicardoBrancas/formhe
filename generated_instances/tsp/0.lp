%formhe-groundtruth:../../examples/tsp.lp

formhe_definition_begin.

place(b). %Berlin
place(d). %Dresden
place(h). %Hamburg
place(l). %Leipzig
place(p). %Potsdam
place(w). %Wolfsburg

link(b, h, 2). link(b, l, 2). link(b, p, 1).
link(d, b, 2). link(d, l, 2). link(d, p, 4).
link(h, b, 2). link(h, l, 2). link(h, w, 3).
link(l, d, 2). link(l, w, 1).
link(p, b, 1). link(p, d, 4). link(p, h, 3).
link(w, d, 2). link(w, h, 3). link(w, l, 1).

formhe_definition_end.

start(X) :- X = #min { Y : place(Y) }.

{ travel(X, Y) : link(X, Y, _) } = 1 :- place(X).

visit(X) :- start(X).
visit(Y) :- visit(X), travel(X, Y).

%:- place(Y), not visit(Y).
:- start(Y), #count { X : travel(X, Y) } < 1.
:- place(Y), #count { X : travel(X, Y) } > 1.

#minimize { C, X : travel(X, Y), link (X, Y, C) }.