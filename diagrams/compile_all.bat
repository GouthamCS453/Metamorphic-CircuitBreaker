@echo off
echo Compiling all LaTeX diagrams...
pdflatex -interaction=nonstopmode class_diagram.tex
pdflatex -interaction=nonstopmode usecase_diagram.tex
pdflatex -interaction=nonstopmode dfd_diagram.tex
pdflatex -interaction=nonstopmode architecture_diagram.tex
echo Done! PDF files are ready.
pause
