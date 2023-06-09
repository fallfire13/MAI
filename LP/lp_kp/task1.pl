father('Евгений Аксёнов', 'Александр Аксёнов').
mother('Анастасия Скворцова', 'Александр Аксёнов').
father('Евгений Аксёнов', 'Елена Аксёнова').
mother('Анастасия Скворцова', 'Елена Аксёнова').
father('Александр Скворцов', 'Анастасия Скворцова').
mother('Валентина Брагина', 'Анастасия Скворцова').
father('Анатолий Аксёнов', 'Евгений Аксёнов').
mother('Татьяна Майорова', 'Евгений Аксёнов').
father('Николай Аксёнов', 'Анатолий Аксёнов').
mother('Татьяна Быкова', 'Анатолий Аксёнов').
father('Николай Аксёнов', 'Людмила Аксёнова').
mother('Татьяна Быкова', 'Людмила Аксёнова').
father('Сергей Майоров', 'Татьяна Майорова').
mother('Екатерина Сазонова', 'Татьяна Майорова').
father('Сергей Майоров', 'Инна Майорова').
mother('Екатерина Сазонова', 'Инна Майорова').
father('Василий Скворцов', 'Александр Скворцов').
mother('Екатерина Добрякова', 'Александр Скворцов').
father('Василий Скворцов', 'Виктор Скворцов').
mother('Екатерина Добрякова', 'Виктор Скворцов').
father('Василий Скворцов', 'Надежда Скворцова').
mother('Екатерина Добрякова', 'Надежда Скворцова').
father('Пётр Брагин', 'Валентина Брагина').
mother('Нина Исавнина', 'Валентина Брагина').
father('Пётр Брагин', 'Галина Брагина').
mother('Нина Исавнина', 'Галина Брагина').
father('Геннадий Муханин', 'Александр Муханин').
mother('Людмила Аксёнова', 'Александр Муханин').

1. Предикат, проверяющий, являются ли два человека братом или сетрой;
```prolog
are_siblings(X, Y):-
    father(Z, X),
    mother(H, X),
    father(Z, Y),
    mother(H, Y),
    X \= Y.
```
2. Предикат, проверяющий, женаты ли два человека;
```prolog
are_married(H, W):-
    father(H, C),
    mother(W, C).
```
3. Предикат, определяюший шурина;
```prolog
brother_in_law(B, X):-
    are_married(X, W),
    mother(W,L),
    are_siblings(B, W),
    father(B, F).
```
4. Предикат, определяюший золовку;
```prolog
sister_in_law(B, X):-
    are_married(X, B),
    father(B,L),
    are_siblings(B, W),
    mother(W, F).
```
Результаты работы предиката
```prolog
?- brother_in_law(X,Y).
X = 'Анатолий Аксёнов',
Y = 'Геннадий Муханин'

?- sister_in_law(X,Y).
X = 'Людмила Аксёнова',
Y = 'Геннадий Муханин'
```