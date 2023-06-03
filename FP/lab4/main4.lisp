(defun single-arg (expr)
  (and (not (null expr)) (not (null (rest expr))) (null (rest (rest expr))))
)

(defun form-to-postfix (expr)
  (if (listp expr)
      (if (single-arg expr)
          (if (eq (first expr) '- )
              (concatenate 'string "0 "(princ-to-string (form-to-postfix (second expr))) " -")
              (concatenate 'string "1 " (princ-to-string (form-to-postfix (second expr))) " /")
          )
          (concatenate 'string (form-to-postfix (second expr)) " "
                               (proc-args (first expr) (rest (rest expr))))
      )
      (string-downcase (princ-to-string expr))
  )
)

(defun proc-args (operator expr)
  (if (not (null expr))
      (if (null (rest expr))
          (concatenate 'string (form-to-postfix (first expr)) " " 
                               (princ-to-string operator)) 
          (concatenate 'string (form-to-postfix (first expr)) " " 
                               (princ-to-string operator) " " 
                               (proc-args operator (rest expr)))
      )
  )
)

(defun t (expr) (form-to-postfix expr))