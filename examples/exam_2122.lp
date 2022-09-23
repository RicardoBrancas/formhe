
formhe_definition_begin.

vertex(a).
vertex(b).
vertex(c).
vertex(d).
vertex(e).

edge(a, b).
edge(b, c).
edge(c, d).
edge(d, a).
edge(e, c).

formhe_definition_end.

1 { vcolor(V, white) ; vcolor(V, black) } 1 :- vertex(V).
:- edge(U, V), vcolor(U, C), vcolor(V, C).