CL-USER 1 : 1 > (setq l1 (make-instance 'line :start (make-instance 'cart :x -5 :y 1) :end (make-instance 'cart :x 6 :y 4)))
[ÎÒÐÅÇÎÊ start [CART x -5 y 1] end [CART x 6 y 4]]

CL-USER 1 : 1 > (line-intersections (list l1))
NIL

CL-USER 3 : 1 > (setq l2 (make-instance 'line :start (make-instance 'cart :x -4 :y 3) :end (make-instance 'cart :x 2 :y 1)))
[ÎÒÐÅÇÎÊ start [CART x -4 y 3] end [CART x 2 y 1]]

CL-USER 4 : 1 > (line-intersections (list l1 l2))
([CART x -23/20 y 41/20])

CL-USER 5 : 1 > (setq l3 (make-instance 'line :start (make-instance 'cart :x 1. :y 4.) :end (make-instance 'cart :x 5. :y 1.)))
[ÎÒÐÅÇÎÊ start [CART x 1 y 4] end [CART x 5 y 1]]

CL-USER 6 : 1 > (line-intersections (list l1 l2 l3))
([CART x -23/20 y 41/20] [CART x 7/3 y 3])

CL-USER 7 : 1 > (setq l4 (make-instance 'line :start (make-instance 'cart :x -5 :y 1) :end (make-instance 'cart :x 5. :y 1.)))
[ÎÒÐÅÇÎÊ start [CART x -5 y 1] end [CART x 5 y 1]]

CL-USER 8 : 1 > (line-intersections (list l1 l2 l3 l4))
([CART x -23/20 y 41/20] [CART x 7/3 y 3] [CART x -5 y 1] [CART x 2 y 1] [CART x 5 y 1])

CL-USER 9 : 1 > (setq l4 (make-instance 'line :start (make-instance 'polar :radius 2 :angle 2) :end (make-instance 'cart :x 5. :y 1.)))
[ÎÒÐÅÇÎÊ start [POLAR r 2 angle 2] end [CART x 5 y 1]]

CL-USER 10 : 1 > (line-intersections (list l1 l2 l3 l4))
([CART x -23/20 y 41/20] [CART x 7/3 y 3] [CART x -1.6022418 y 1.9266611] [CART x -0.18194358 y 1.7273145])