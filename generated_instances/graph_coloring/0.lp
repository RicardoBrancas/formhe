%formhe-groundtruth:../../examples/graph_coloring.lp
#program base.
#show is_color/2.
country(belgium;denmark;france;germany;netherlands;luxembourg).
color(red;green;blue;white).
arc(france,belgium;france,luxembourg;france,germany).
arc(luxembourg,germany;luxembourg,belgium).
arc(netherlands,belgium).
arc(germany,belgium;germany,netherlands;germany,denmark).
neighbor(X,Y) :- arc(X,Y).
neighbor(Y,X) :- arc(X,Y).
1 <= { is_color(X,C): color(C) } <= 1 :- country(X).
% selected out statement:
%#false :- is_color(X1,C); is_color(X2,C); neighbor(X1,X2).
#false :- is_color(germany,red).
#false :- is_color(france,blue).