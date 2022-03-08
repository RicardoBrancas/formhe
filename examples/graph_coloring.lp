country(belgium;denmark;france;germany;netherlands;luxembourg).
% 3 color is not enough
% color(red;green;blue).
color(red;green;blue;white).
arc(france,belgium;france,luxembourg;france,germany).
arc(luxembourg,germany;luxembourg,belgium).
arc(netherlands,belgium).
arc(germany,belgium;germany,netherlands;germany,denmark).
neighbor(X,Y) :- arc(X,Y).
neighbor(Y,X) :- arc(X,Y).
% Ensure that each country has exactly one color,
1 {is_color(X, C) : color(C) } 1 :- country(X).
% Two neighboring countries cannot have the same color.
:- is_color(X1, C), is_color(X2, C), neighbor(X1,X2).
% symmetry breaking
:- is_color(germany, red).
:- is_color(france, blue).
#show is_color/2.