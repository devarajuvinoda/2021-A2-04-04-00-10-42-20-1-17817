# Deep learning for automatically fixing single-line syntax errors in C programs

The goal of this assignment is to design, implement and evaluate a deep learning based solution to fix syntactic errors in C programs. The programs were written by students in introductory programming assignments and contain single-line errors.

The model is trained to perform **Line-to-line fixing** in which the model will map token sequence of only the buggy line in the input program (sourceLineText) to the corresponding fixed line (targetLineText).
