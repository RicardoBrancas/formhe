%formhe-problem:A
%formhe-timestamp:1699275399
%formhe-selfeval-lines:0
%formhe-selfeval-fix:v(V) :- e(V, _).

v(V) :- assign(V, _).
v(V) :- e(_, V).
1 { assign(V, (1..k)) } 1 :- v(V).
 :- e(U, V), assign(U, C), assign(V, C).