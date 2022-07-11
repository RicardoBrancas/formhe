%%writefile mythological.lp
1 {mythical(X); -mythical(X)} 1 :- animal(X).
1 {horned(X); -horned(X)} 1 :- animal(X).
1 {magical(X); -magical(X)} 1 :- animal(X).
1 {mammal(X); -mammal(X)} 1 :- animal(X).
1 {immortal(X); -immortal(X)} 1 :- animal(X).

immortal(X) :- mythical(X).
-immortal(X) :- -mythical(X).
mammal(X) :- -mythical(X).

horned(X) :- immortal(X).
horned(X) :- mammal(X).
magical(X) :- horned(X).

animal(unicorn).