# Отчет по лабораторной работе №1
## Работа со списками и реляционным представлением данных
## по курсу "Логическое программирование"

### студент: Аксенов А.Е.

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |
| Левинская М.А.|              |               |

## Введение
Списки на языке Пролог отличаются от спиков в большинстве императивных языков. Например, в C++ список может хранить только однотипные значения, а конкретный элемент списка можно получить с помощью итератора. В языке Пролог список может хранить разнотипные значения, а получить конкретный элемент можно с помощью унификации и оператора отсечения головы.

Список в Прологе напоминает стек, так как проще всего работать с первым элементом списка, нежели с остальными.

## Задание 1.1: Предикат обработки списка
Удаление последнего элемента списка.

Без стандартных предикатов реализуем следующим образом:
Если список состоит из одного элемента, возвращаем пустой список.
Если список состоит из нескольких элементов, удаляем последний элемент у хвоста списка.

```prolog
remLast1([_], []).
remLast1([X|T], [X|T1]) :- remLast1(T, T1).
```

Со стандартными предикатами реализуем следующим образом:
Находим список L, такой, что если к нему добавить какой-нибудь элемент в конец, то получится исходный список.

```prolog
remLast2(L, L1) :- myAppend(L1, [_], L).
```

Пример работы:

```prolog
?- remLast1([], X).
false.

?- remLast1([1], X).
X = [] ;
false.

?- remLast1([1,2,3], X).
X = [1, 2] ;
false.

?- remLast2([], X).
false.

?- remLast2([1], X).
X = [] ;
false.

?- remLast2([1,2,3], X).
X = [1, 2] ;
false.
```

## Задание 1.2: Предикат обработки числового списка
Проверка списка на упорядоченность

Без стандартных предикатов реализуем следующим образом:
Если список пустой или состоит из одного элемента, то он автоматически упорядочен.
Если список состоит из нескольких элементов, то проверяем, что первый элемент меньше второго и проверяем хвост на упорядоченность.

```prolog
order1([]).
order1([_]).
order1([X,Y|T]) :- X < Y, order1([Y|T]).
```

Со стандартными предикатами реализуем следующим образом:
Если в списке есть подсписок из двух элементов, такой что первый его элемент больше второго, то этот список не упорядочен.

```prolog
order2(L) :- mySublist(L, [X,Y]), X>Y, !, false.
order2(_).
```

Пример работы:

```prolog
?- order1([]).
true.

?- order1([1]).
true ;
false.

?- order1([1,2,3]).
true ;
false.

?- order1([1,3,2]).
false.

?- order2([]).
true.

?- order2([1]).
true.

?- order2([1,2,3]).
true.

?- order2([1,3,2]).
false.
```

## Задание 2: Реляционное представление данных
Главным преимуществом реляционного представления является то, что не нужно задавать какую-либо логику и закономерности данных. Мы указываем все известные данные в явном виде. Это значительно проще, чем пытаться найти закономерности в записываемых данных. Это также позволяет быстро проверить истинность утверждения, напрямую связанного с данными, без долгих вычислений, так как всё задано в явном виде.

Преимущество представления 'two.pl' в том, что все данные представлены одним предикатом. С помощью него мы можем получить любую необходимую информацию. Недостатком данного представления является то, что очень много повторений данных. То, что Петров учится в 102 группе записано 6 раз. Можно было бы хранить данные экономнее, если использовать несколько предикатов, а не один.

### Решение:

Чтобы использовать предикаты из файла `two.pl`, ипортируем его в начале файла:

```prolog
:- ['two.pl'].
```

1) Напечатать средний балл для каждого предмета

Сначала вычислим средний балл для одного предмета:

```prolog
avgGrade(Subj, Avg) :-
    findall(X, grade(_, _, Subj, X), L),
    sum(L, Sum), length(L, Len), Avg is Sum / Len.

sum([X|T], S) :- sum(T, S1), S is X + S1.
sum([], 0).
```

Напишем вспомогательный предикат, который получает на вход список предметов, и для каждого из них печатает средний балл:

```prolog
printAvgList([X|T]) :-
    avgGrade(X, A), write(X), write(': '), write(A), nl, printAvgList(T).
printAvgList([]).
```

Чтобы решить задачу, надо сгенерировать список предметов и с помощью предыдущего предиката напечатать средние баллы. Список предметов можно получить комбинацией findall и setof:

```prolog
task1() :-
    findall(X, grade(_, _, X, _), S1),
    setof(X, member(X, S1), S),
    printAvgList(S).
```

Результат:

```prolog
?- task1.
Английский язык: 3.75
Информатика: 3.9285714285714284
Логическое программирование: 3.9642857142857144
Математический анализ: 3.892857142857143
Психология: 3.9285714285714284
Функциональное программирование: 3.9642857142857144
true
```

2) Для каждой группы, найти количество не сдавших студентов

Данная задача решается аналогично. Сначала реализуем предикат, находящий число несдавших студентов в группе:

```prolog
numFail(G, N) :-
    findall(X, grade(G, X, _, 2), L),
    length(L, N).
```

Затем вспомогательный предикат, печатающий для списка групп:

```prolog
printFailList([X|T]) :-
    numFail(X, Y), write(X), write(': '), write(Y), nl, printFailList(T).
printFailList([]).
```

И затем, генерируем список групп по такому же принципу:

```prolog
task2() :-
    findall(X, grade(X, _, _, _), G1),
    setof(X, member(X, G1), G),
    printFailList(G).
```

Результат:

```prolog
?- task2.
101: 2
102: 5
103: 4
104: 2
true
```

3) Найти количество не сдавших студентов для каждого из предметов

В данной задаче нужно сделать почти то же самое, что и в предыдущей. Потому немного модифицируем предикаты из предыдущей задачи:

```prolog
numFailSubj(S, N) :-
    findall(X, grade(_, X, S, 2), L),
    length(L, N).

printFailSubjList([X|T]) :-
    numFailSubj(X, Y), write(X), write(': '), write(Y), nl, printFailSubjList(T).
printFailSubjList([]).

task3() :-
    findall(X, grade(_, _, X, _), S1),
    setof(X, member(X, S1), S),
    printFailSubjList(S).
```

Результат:

```prolog
?- task3.
Английский язык: 4
Информатика: 2
Логическое программирование: 2
Математический анализ: 3
Психология: 1
Функциональное программирование: 1
true
```

## Выводы
Лабораторная работа научила меня основам логического программирования и работе со списками в частности. Я научился создавать предикаты для решения простейших задач на Прологе, такими как обработка списков и работа с данными в реляционном представлении. 
Программы на Прологе выглядят простыми, но за ними скрывается мощный логический бэкграунд. В этой работе у нам представилась возможность познакомиться с простыми задачами на prolog, что дает нам базу для решения более сложных заданий в будущем.