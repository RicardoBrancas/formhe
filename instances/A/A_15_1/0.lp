%formhe-problem:A
%formhe-timestamp:1699275399
%formhe-selfeval-lines:2
%formhe-selfeval-fix:1 { assign(V, (1..k)) } 1 :- v(V).

v(V) :- e(_, V).
v(V) :- e(V, _).
1 { assign(V, (1..k)) }  :- v(V).
 :- e(U, V), assign(U, C), assign(V, C).