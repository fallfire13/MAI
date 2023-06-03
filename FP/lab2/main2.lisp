(defun remove-last(list)
  (cond ((null (cdr list))
     nil)
    (t
     (cons (car list)
           (remove-last (cdr list))))))