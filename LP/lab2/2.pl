remove([X|T], X, T).
remove([H|T], X, [H|T1]) :- remove(T, X, T1).

permute([], []).
permute(L, L3) :- remove(L, X, L1), permute(L1, L2), append([X], L2, L3).

solve(Dancer, Artist, Singer, Writer) :-
    permute([Dancer, Artist, Singer, Writer], [voronov, pavlov, levickiy, saharov]),
    Singer \= voronov, Singer \= levickiy,
    Artist \= pavlov, Writer \= pavlov,
    Writer \= saharov, Writer \= voronov,
    not((Artist = voronov, Writer = levickiy)),
    not((Writer = voronov, Artist = levickiy)).
