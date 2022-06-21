house(1..3).

1 {blue(X); green(X); white(X)} 1 :- house(X).
1 {blue(X) : house(X)} 1.
1 {green(X) : house(X)} 1.
1 {white(X) : house(X)} 1.

1 {dutch(X); german(X); italian(X)} 1 :- house(X).
1 {dutch(X) : house(X)} 1.
1 {german(X) : house(X)} 1.
1 {italian(X) : house(X)} 1.

1 {coffee(X); tea(X); water(X)} 1 :- house(X).
1 {coffee(X) : house(X)} 1.
1 {tea(X) : house(X)} 1.
1 {water(X) : house(X)} 1.

% There is one house between the house of the person drinking coffee and the blue house
:- coffee(X), blue(Y), |X - Y| < 2.

% The person drinking water lives in the blue house
water(X) :- blue(X), house(X).

% The Italian lives to the left of the coffee drinking person
1 {italian(Y) : house(Y), Y < X} :- coffee(X), house(X).

green(3).
german(2).