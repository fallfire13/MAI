remove([X|T], X, T).
remove([H|T], X, [H|T1]) :- remove(T, X, T1).

%состояние: s(число миссионеров на левом берегу, число людоедов на левом берегу,
%число миссионеров на правом берегу, число людоедов на правом берегу, положение лодки)

%числа, сумма которых не меньше 1 и не больше 3 и первое не меньше второго
boat(0, 1).
boat(0, 2).
boat(0, 3).
boat(1, 0).
boat(1, 1).
boat(2, 0).
boat(2, 1).
boat(3, 0).

%теоритически возможный переход
possible(s(ML, LL, MR, LR, left), s(ML1, LL1, MR1, LR1, right)) :-
    boat(DM, DL), ML1 is ML-DM, MR1 is MR+DM, LL1 is LL-DL, LR1 is LR+DL.

possible(s(ML, LL, MR, LR, right), s(ML1, LL1, MR1, LR1, left)) :-
    boat(DM, DL), ML1 is ML+DM, MR1 is MR-DM, LL1 is LL+DL, LR1 is LR-DL.

%проверка условий
goodState(s(ML, LL, MR, LR, _)) :-
    ML >= 0, LL >= 0, MR >= 0, LR >= 0, (ML >= LL, !; ML = 0), (MR >= LR, !; MR = 0).

%переход, не нарушающий правила
move(X, Y) :- possible(X, Y), goodState(Y).

%prolong(Path, NewPath)
prolong([S|T], [S1,S|T]) :- move(S, S1), not(member(S1, T)).

%преобразовать путь в последовательность действий
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

describe([_], []).
describe([X,Y|T], A) :- describe([Y|T], A1), descMove(Y, X, A2), append(A1, [A2], A).

%алгоритмы поиска
%deep(Path, End, NewPath)
deep([End|T], End, [End|T]).
deep(Path, End, NewPath) :- prolong(Path, Path1), deep(Path1, End, NewPath).

deep(Path) :- deep([s(3,3,0,0,left)], s(0,0,3,3,right), Path).

%breath(Pathes, End, NewPathes)
breath([[End|T]|_], End, [End|T]).
breath([Path|T], End, NewPathes) :- findall(X, prolong(Path, X), L), !,
    append(T, L, P1), breath(P1, End, NewPathes).
breath([_|T], End, NewPathes) :- breath(T, End, NewPathes).

breath(Path) :- breath([[s(3,3,0,0,left)]], s(0,0,3,3,right), Path).

%ideep(Path, End, NewPath, Depth)
ideep([End|T], End, [End|T], 0).
ideep(Path, End, NewPath, Depth) :- Depth > 0, D1 is Depth - 1,
    prolong(Path, Path1), ideep(Path1, End, NewPath, D1).

generate(0, _).
generate(X, A) :- A > 0, A1 is A-1, generate(Y, A1), X is Y+1.

ideep(Path) :- generate(X, 100), ideep([s(3,3,0,0,left)], s(0,0,3,3,right), Path, X).

%хронометрия
duration(F, Time) :-
    get_time(Start), call(F), get_time(End), Time is End - Start, !.
