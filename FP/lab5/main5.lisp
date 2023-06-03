(defclass cart() ; точка в декартовых координатах
  (
   (x :initarg :x :accessor cart-x)
   (y :initarg :y :accessor cart-y)
  )
)

(defmethod print-object ((c cart) stream) ; распечатка
  (format stream "[CART x ~d y ~d]" (cart-x c) (cart-y c) )
)

(defclass polar () ; точка в пол€рных координатах
 ((radius :initarg :radius :accessor radius) ; радиус >=0
  (angle :initarg :angle :accessor angle))   ; угол (-pi;pi]
)

(defmethod print-object ((p polar) stream) ; распечатка
  (format stream "[POLAR r ~d angle ~d]" (radius p) (angle p))
)

(defmethod cart-x ((p polar))
  (* (radius p) (cos (angle p)))
)

(defmethod cart-y ((p polar))
  (* (radius p) (sin (angle p)))
)

(defclass line () ; отрезок
  ((start :initarg :start :accessor line-start)
   (end :initarg :end :accessor line-end))
)

(defmethod print-object ((lin line) stream)
  (format stream "[ќ“–≈«ќ  start ~s end ~s]" (line-start lin) (line-end lin))
)

(defun solve2 (a11 a12 a21 a22 b1 b2)
  (let ((d  (- (* a11 a22) (* a12 a21)))
        (dx (- (* b1 a22) (* b2 a12)))
        (dy (- (* b2 a11) (* b1 a21)))
       )
   (list (/ dx d) (/ dy d))
  )
)

(defun kb (line)
  (solve2 (cart-x (line-start line)) 1 (cart-x (line-end line)) 1 (cart-y (line-start line)) (cart-y (line-end line)))
)

(defun intersection-point (line1 line2)
  (let* (
         (kb1 (kb line1))
         (kb2 (kb line2))
         (xy  (solve2 (nth 0 kb1) -1 (nth 0 kb2) -1 (- (nth 1 kb1)) (- (nth 1 kb2))))
         (x1 (cart-x (line-start line1)))
         (y1 (cart-y (line-start line1)))
         (x2 (cart-x (line-end line1)))
         (y2 (cart-y (line-end line1)))
         (x3 (cart-x (line-start line2)))
         (y3 (cart-y (line-start line2)))
         (x4 (cart-x (line-end line2)))
         (y4 (cart-y (line-end line2)))
         (st  (solve2 (- x2 x1) (- x3 x4) (- y2 y1) (- y3 y4) (- x3 x1) (- y3 y1)))
       )
    (if (and (<= 0 (nth 0 st)) (<= (nth 1 st) 1)) xy nil)
  )
)

(defun point-exists (point plist)
  (let ((find nil))
    (loop for i from 0 to (- (list-length plist) 1) do
      (when (and (= (cart-x point) (cart-x (nth i plist))) 
                 (= (cart-y point) (cart-y (nth i plist))))
        (setq find T) (return find)
      )
    )
    find
  )
)

(defun line-intersections (lines)
  (let ((res ()))
     (loop for i from 0 to (- (list-length lines) 2) do
        (loop for j from (+ i 1) to (- (list-length lines) 1) do
            (let* (
                   (point (intersection-point (nth i lines) (nth j lines)))
                   (cpoint (if point (make-instance 'cart :x (nth 0 point) :y (nth 1 point)) nil))
                  )
              (when (and cpoint (not (point-exists cpoint res))) (setq res (append res (list cpoint)))) ; исключаем дубликаты
            )
         )
     )
    res
  )
)

