%стандартные предикаты
myLength([], 0).
myLength([_|T], N) :- myLength(T, N1), N is N1+1.

myMember([X|_], X).
myMember([_|T], X) :- myMember(T, X).

myAppend([], X, X).
myAppend([H|T], X, [H|T1]) :- myAppend(T, X, T1).

myRemove([X|T], X, T).
myRemove([H|T], X, [H|T1]) :- myRemove(T, X, T1).

myPermute([], []).
myPermute(L, L3) :- myRemove(L, X, L1), myPermute(L1, L2), myAppend([X], L2, L3).

mySublist(L, S) :- myAppend(_, S1, L), myAppend(S, _, S1).

%удаление последнего элемента списка
remLast1([_], []).
remLast1([X|T], [X|T1]) :- remLast1(T, T1).

remLast2(L, L1) :- myAppend(L1, [_], L).

%чпроверка списка на упорядоченность
order1([]).
order1([_]).
order1([X,Y|T]) :- X < Y, order1([Y|T]).

order2(L) :- mySublist(L, [X,Y]), X>Y, !, false.
order2(_).
