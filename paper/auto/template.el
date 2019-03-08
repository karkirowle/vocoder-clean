(TeX-add-style-hook
 "template"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "a4paper")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "url")
   (TeX-run-style-hooks
    "latex2e"
    "init_pos"
    "blstm_act"
    "retraining"
    "retraining_linear"
    "article"
    "art10"
    "INTERSPEECH2018"
    "tikz"
    "url"
    "blindtext")
   (LaTeX-add-labels
    "tab:electrodes"
    "fig:mask"
    "fig:structure"
    "tab:example"
    "nnexperiment"
    "tab:all_data"
    "tab:transfer"
    "tab:pilot"
    "tab:architectures"
    "eq1"
    "eq2"
    "eq3"
    "eq4"
    "tab:example1"
    "learning_curve"
    "retraining_linear")
   (LaTeX-add-bibliographies
    "paper1"))
 :latex)

