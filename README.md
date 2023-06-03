# thesis-NTNU

CoPCSE@NTNU – Community of Practice for Computer Science Education at the Norwegian University of Science and Technology – is an informal forum for lecturers in computer science and related fields across campuses and departments.

The current repository provides a LaTeX thesis template that should in principle be applicable for theses at all study levels – bachelor, master and PhD. It is closely based on the standard LaTeX report document class with added packages and customisations. The purpose of the document provided in `thesis.tex` is threefold. It should serve (i) as a description of the document class, (ii) as an example of how to use it, and (iii) as a thesis template.

The template does not have any official status, and it is not a general NTNU-level requirement to use it. It replaces previous templates like https://github.com/COPCSE-NTNU/bachelor-thesis-NTNU and https://github.com/COPCSE-NTNU/master-theses-NTNU.

## Setting up

You can use the template with [Overleaf](http://overleaf.com), and you are strongly encouraged to do so. The alternative is to install local copy of LaTeX on your laptop (not adviced, huge, difficult).

You should **fork** the CoPCSE repo so that you have your own files to edit and you can always merge with the upstream changes to the template, in case the template is updated. 

### Setup using Overleaf

There are two ways for setting up the [**Overleaf**](http://overleaf.com) project with the template:

* Use the `.zip` copy and upload.
* Fork the the CoPCSE repo so that you have your own files to edit.

### Building document locally

The template also provides a simple `Makefile` which allows you to build the document locally. This requires that you have a LaTeX compiler, such as [`texlive`](https://www.tug.org/texlive/), installed locally, which has to provide the commands `pdflatex` and `biber`.

\begin{figure}[htbp]
  \centering
  \begin{tikzpicture}[
      node distance=1cm,
      process/.style={rectangle, rounded corners, minimum width=4cm, minimum height=1.5cm, text width=3.6cm, align=center, draw=black},
      process1/.style={process, fill=gray!20},
      process2/.style={process, fill=gray!50},
      arrow/.style={thick,->,>=stealth, line width=1.5pt, draw=gray}
    ]

    \node (step1) [process1] {Generate Random Architectures};
    \node (step2) [process2, below=of step1] {Conduct Comprehensive Training on Architectures for Benchmarking};
    \node (step3) [process1, below=of step2] {Establish Zero-Cost Proxy Framework};
    \node (step4) [process2, below=of step3] {Collect Data on Zero-Cost Proxies Performance};
    \node (step5) [process1, below=of step4] {Perform Quantitative Analysis Using Collected Data};

    \draw [arrow] (step1) -- (step2);
    \draw [arrow] (step2) -- (step3);
    \draw [arrow] (step3) -- (step4);
    \draw [arrow] (step4) -- (step5);
  \end{tikzpicture}
  \caption{Flowchart illustrating the research process.}
  \label{fig:research_flowchart}
\end{figure}
