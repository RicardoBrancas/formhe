#const total = 1505.
#const n = 10.
amount(0..n).
food(mixed_fruit;french_fries;side_salad;hot_wings;mozzarella_sticks;samples_place).
price(mixed_fruit,215).
price(french_fries,275).
price(side_salad,335).
price(hot_wings,355).
price(mozzarella_sticks,420).
price(samples_place,580).
prices(P) :- price(_, P).
% each food has exactly one amount
1 { food_amount(Food, Amount) : amount(Amount) } 1 :- food(Food).

% prices sums to total
total = #sum{ Price*Amount,F:food_amount(F, Amount) : price(F, Price), prices(Price), amount(Amount) }.
#show food_amount/2.