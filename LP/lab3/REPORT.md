# № Отчет по лабораторной работе №3
## по курсу "Логическое программирование"

## Решение задач методом поиска в пространстве состояний

### студент: Аксенов А.Е.

## Результат проверки

| Преподаватель     | Дата         |  Оценка       |
|-------------------|--------------|---------------|
| Сошников Д.В. |              |               |
| Левинская М.А.|              |               |

## Введение
Удобнее всего из задач на поиск решаются задачи с малым числом состояний. В данном случае любой алгоритм поиска занимает мало времени на рассчёты. При большом количестве состояний эффективнее всего работает поиск в глубину, но он занимает иного памяти. Итеративный поиск работает примерно в 2 раза медленнее, но зато не расходует много памяти. Обычный поиск в глубину становится неэффективным. При бесконечном числе состояний поиск в глубину невозможно использовать, так как он может зациклиться.

Алгоритмы поиска легко преобразуются в рекурсию, потому их легко реализовать на Прологе. К тому же благодаря бэктрекингу мы сможем найти все пути, а не только первый найденный.

## Задание
Три миссионера и три каннибала хотят переправиться с левого берега реки на правый. Как это сделать за минимальное число шагов, если в их распоряжении имеется трехместная лодка и ни при каких обстоятельствах (в лодке или на берегу) миссионеры не должны оставаться в меньшинстве.

## Принцип решения
Состояние представим в виде терма s, хранящего число миссионеров на левом берегу, число людоедов на левом берегу, число миссионеров на правом берегу, число людоедов на правом берегу и положение лодки.

### Пример:
Начальное состояние: `s(3, 3, 0, 0, left)`.

На лодке должен быть хотя бы один человек, но не больше трёх, и если на лодке есть миссионеры, их должно быть не меньше, чем людоедов. Такие комбинации расположений людей в лодке описаны в предикате boat.

```prolog
boat(0, 1).
boat(0, 2).
boat(0, 3).
boat(1, 0).
boat(1, 1).
boat(2, 0).
boat(2, 1).
boat(3, 0).
```

Реализуем предикат `possible`, который генерирует новые возможные состояния без проверки их на удовлетворение условий задачи.

```prolog
possible(s(ML, LL, MR, LR, left), s(ML1, LL1, MR1, LR1, right)) :-
    boat(DM, DL), ML1 is ML-DM, MR1 is MR+DM, LL1 is LL-DL, LR1 is LR+DL.
possible(s(ML, LL, MR, LR, right), s(ML1, LL1, MR1, LR1, left)) :-
    boat(DM, DL), ML1 is ML+DM, MR1 is MR-DM, LL1 is LL+DL, LR1 is LR-DL.
```

Предикат `goodState` проверяет, чтобы и на левом берегу и на правом берегу миссионеры не были в меньшинстве, и чтобы их число не было отрицательным.

```prolog
goodState(s(ML, LL, MR, LR, _)) :-
    ML >= 0, LL >= 0, MR >= 0, LR >= 0, (ML >= LL, !; ML = 0), (MR >= LR, !; MR = 0).
```

С помощью этих предикатов реализуем предикат move перехода из одного состояния в другое, не нарушающее правила задачи:

```prolog
move(X, Y) :- possible(X, Y), goodState(Y).
```

Наконец, реализуем предикат `prolong`. Данный предикат генерирует все возможные способы продлевания пути. Ранее посещённые состояния при этом не используются. Этот предикат лежит в основе всех алгоритмов поиска.

```prolog
prolong([S|T], [S1,S|T]) :- move(S, S1), not(member(S1, T)).
```

Кроме самого пути, нам потребуется список текстовых описаний переходов. Для генерации данного списка по заданному пути используется следующий предикат:

```prolog
describe([_], []).
describe([X,Y|T], A) :- describe([Y|T], A1), descMove(Y, X, A2), append(A1, [A2], A).
```

Он использует вспомогательные предикаты для генерации текстовых описаний:

```prolog
descDiff(0, _, '').
descDiff(1, Who, A) :- atom_concat('1 ', Who, A).
descDiff(D, Who, A) :- D > 1, atom_concat(D, ' ', A1),
    atom_concat(A1, Who, A2), atom_concat(A2, 's', A).

descConcat('', X, X).
descConcat(X, '', X).
descConcat(X, Y, A) :- X \= '', Y \= '', atom_concat(X, ' and ', A1), atom_concat(A1, Y, A).

descMove(s(ML, LL, _, _, left), s(ML1, LL1, _, _, right), A) :-
    DM is ML - ML1, DL is LL - LL1,
    descDiff(DM, 'missioner', AM), descDiff(DL, 'cannibal', AL),
    descConcat(AM, AL, A1), atom_concat(A1, ' swims to the right coast', A).

descMove(s(_, _, MR, LR, right), s(_, _, MR1, LR1, left), A) :-
    DM is MR - MR1, DL is LR - LR1,
    descDiff(DM, 'missioner', AM), descDiff(DL, 'cannibal', AL),
    descConcat(AM, AL, A1), atom_concat(A1, ' swims to the left coast', A).
```

## Поиск в глубину:
Данный поиск реализуется проще всего. Мы рекурсивно вызываем prolong, получая тем все возможные пути.

```prolog
deep([End|T], End, [End|T]).
deep(Path, End, NewPath) :- prolong(Path, Path1), deep(Path1, End, NewPath).

deep(Path) :- deep([s(3,3,0,0,left)], s(0,0,3,3,right), Path).
```

### Пример использования:

```prolog
?- deep(_Path), describe(_Path, Text).
Text = ['2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 missioner and 1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast'];
Text = ['2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 missioner and 1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 missioner swims to the left coast', '1 missioner and 1 cannibal swims to the right coast'];
Text = ['2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 missioner and 1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 cannibal swims to the left coast', '3 cannibals swims to the right coast'];
...
Text = ['1 missioner and 1 cannibal swims to the right coast', '1 missioner swims to the left coast', '3 missioners swims to the right coast', '2 missioners swims to the left coast', '2 missioners and 1 cannibal swims to the right coast', '1 missioner swims to the left coast', '1 missioner and 1 cannibal swims to the right coast'];
false.
```

## Поиск в ширину:
Вместо пути мы храним очередь из путей. Из очереди извлекается первый путь, продлевается всеми возможными способами с помощью `findall` и полученные пути добавляются в конец очереди.

```prolog
breath([[End|T]|_], End, [End|T]).
breath([Path|T], End, NewPathes) :- findall(X, prolong(Path, X), L), !,
    append(T, L, P1), breath(P1, End, NewPathes).
breath([_|T], End, NewPathes) :- breath(T, End, NewPathes).

breath(Path) :- breath([[s(3,3,0,0,left)]], s(0,0,3,3,right), Path).
```

### Пример использования:

```
?- breath(_Path), describe(_Path, Text).
Text = ['2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '3 missioners swims to the right coast', '1 cannibal swims to the left coast', '3 cannibals swims to the right coast'];
Text = ['3 cannibals swims to the right coast', '1 cannibal swims to the left coast', '3 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast'];
...
Text = ['1 missioner and 1 cannibal swims to the right coast', '1 missioner swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 missioner and 1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 missioner swims to the left coast', '1 missioner and 1 cannibal swims to the right coast'];
false.
```

## Итерационный поиск в глубину:
Реализуем предикат, генирирующий числа от 0 до A:

```prolog
generate(0, _).
generate(X, A) :- A > 0, A1 is A-1, generate(Y, A1), X is Y+1.
```

Модифицируем поиск в глубину, добавив дополнительный параметр. Теперь предикат истинен только если этот параметр достигает нуля. При каждом углублении данный параметр уменьшается на единицу.

```prolog
ideep([End|T], End, [End|T], 0).
ideep(Path, End, NewPath, Depth) :- Depth > 0, D1 is Depth - 1,
    prolong(Path, Path1), ideep(Path1, End, NewPath, D1).
```

Итоговый предикат имеет следющий вид:

```prolog
ideep(Path) :- generate(X, 100), ideep([s(3,3,0,0,left)], s(0,0,3,3,right), Path, X).
```

### Пример использования:

```prolog
?- iteration(Path).
Text = ['2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '3 missioners swims to the right coast', '1 cannibal swims to the left coast', '3 cannibals swims to the right coast'];
Text = ['3 cannibals swims to the right coast', '1 cannibal swims to the left coast', '3 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast'];
...
Text = ['1 missioner and 1 cannibal swims to the right coast', '1 missioner swims to the left coast', '2 cannibals swims to the right coast', '1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 missioner and 1 cannibal swims to the left coast', '2 missioners swims to the right coast', '1 cannibal swims to the left coast', '2 cannibals swims to the right coast', '1 missioner swims to the left coast', '1 missioner and 1 cannibal swims to the right coast'];
false.
```

## Результаты
Найдём длины первых путей, найденных алгоритмами:

```prolog
?- deep(_Path), length(_Path, N).
N = 12.
?- breath(_Path), length(_Path, N).
N = 6.
?- ideep(_Path), length(_Path, N).
N = 6.
```

Для измерения времени введём предикат `duration(F, Time)`. Данный предикат вычисляет, сколько времени занимает выполнение предиката `F` в миллисекундах.

```prolog
duration(F, Time) :-
    get_time(Start), call(F), get_time(End), Time is End - Start, !.
```

С помощью него вычисляем время работы алгоритмов:

```prolog
?- duration(deep(_), X).
X = 0.00011992454528808594.
?- duration(breath(_), X).
X = 0.0016798973083496094.
?- duration(ideep(_), X).
X = 0.0007798671722412109.
```

| Алгоритм поиска | Длина найденного первым пути | Время работы              |
|-----------------|------------------------------|---------------------------|
| В глубину       | 12 состояний                 | 0.00011992454528808594 ms |
| В ширину        | 6 состояний                  | 0.0016798973083496094 ms  |
| ID              | 6 состояний                  | 0.0007798671722412109 ms  |

## Выводы
В ходе лабораторной работы была успешно выполнена поставленная цель изучен метод поиска в пространстве состояний для решения задачи. Были реализованы три алгоритма поиска в графах: в ширину, глубину и с итеративным углублением.

Из результата работы программы мы видим, что самым быстрым является DFS. Но он нашёл не кратчайший путь. Поиск в ширину и ID нашли кратчайший путь. Но поиск в ширину занял более чем в два раза больше времени. Поэтому в данной задаче Id является самым оптимальным алгоритмом.
