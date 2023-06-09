parent(alexei,tolia).
parent(alexei,volodia).
parent(alexei,dima).
parent(tolia,tima).

brother(X,Y) :- parent(Z,X), parent(Z,Y), X\=Y.

brotherYN(X, Y, yes) :- brother(X, Y), !.
brotherYN(_, _, no).

%контекст
%родительный
context(alexei, 'rod', alexeia).
context(tolia, 'rod', toli).
context(volodia, 'rod', volodi).
context(dima, 'rod', dimi).
context(tima, 'rod', timi).
%притяжательный
context(alexei, 'prit', alexein).
context(tolia, 'prit', tolin).
context(volodia, 'prit', volodin).
context(dima, 'prit', dimin).
context(tima, 'prit', timin).

%грамматика (виды вопросов):
%Q = IP 'brat' RP '?'
%Q = 'kto' PP 'brat' '?'
%Q = 'chei' 'brat' IP '?'
%Q = 'kto' 'brat' RP '?'
%IP - именительный падеж
%RP - родительный падеж
%PP -  притяжательная форма

answer([X, brat, Y, '?'], Z) :-
    context(Y1, 'rod', Y), brotherYN(X, Y1, Z).

answer([kto, X, brat, '?'], Y) :-
    context(X1, 'prit', X), brother(X1, Y).

answer([chei, brat, X, '?'], Y) :-
    brother(X, Y).

answer([kto, brat, X, '?'], Y) :-
    context(X1, 'rod', X), brother(X1, Y).
