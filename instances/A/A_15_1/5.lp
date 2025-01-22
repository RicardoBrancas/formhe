%formhe-problem:A
%formhe-timestamp:1699275399
%formhe-selfeval-lines:0 1
%formhe-selfeval-fix: :- e(U, V), assign(U, C), assign(V, C). v(V) :- e(V, _).

 :- e(U, V), assign(U, C), assign(U, C).
v(V) :- e(V, C).
v(V) :- e(_, V).
1 { assign(V, (1..k)) } 1 :- v(V).