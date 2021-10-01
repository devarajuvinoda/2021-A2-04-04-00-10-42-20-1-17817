# Deep learning for automatically fixing single-line syntax errors in C programs

* The goal of this project is to design, implement and evaluate a deep learning based solution to fix syntactic errors in C programs. 
* The dataset consisits of programs that were written by students in introductory programming assignments and contain single-line errors.

The model is trained to perform **Line-to-line fixing** in which the model will map token sequence of only the buggy line in the input program (sourceLineText) to the corresponding fixed line (targetLineText).

Reference:
* [Effective Approaches to Attention-based Neural Machine Translation.](https://arxiv.org/abs/1508.04025v5)
