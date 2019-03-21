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
    "retraining_lc"
    "article"
    "art10"
    "INTERSPEECH2018"
    "tikz"
    "url"
    "blindtext")
   (LaTeX-add-labels
    "section:method"
    "tab:electrodes"
    "fig:electrodes"
    "fig:mask"
    "fig:structure"
    "tab:example"
    "section:nnexperiment"
    "tab:all_data"
    "tab:architectures"
    "section:speech"
    "section:visualisation"
    "section:limitations"
    "learning_curve")
   (LaTeX-add-bibliographies
    "paper1"))
 :latex)

