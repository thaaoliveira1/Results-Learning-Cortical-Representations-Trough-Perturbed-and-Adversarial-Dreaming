\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{midl}[2020/10/18 v1.3 MIDL Proceedings]

% Note that if the use of NoHyper below still causes
% issues, we could suppress hyperlinks to footnotes entirely
% \PassOptionsToPackage{hyperfootnotes=false}{hyperref}

\@ifundefined{ifanonsubmission}{
  \newif\ifanonsubmission
  \anonsubmissionfalse
}{}

\@ifundefined{iffinalsubmission}{
  \newif\iffinalsubmission
  \finalsubmissionfalse
}{}

\DeclareOption{anon}{\anonsubmissiontrue}
\DeclareOption{final}{\finalsubmissiontrue}
\ProcessOptions\relax

\LoadClass[pmlr]{jmlr}

\@ifclasslater{jmlr}{2020/01/17}{%
% Using a post 1.25 version of jmlr.cls - all good
}{%
% Temporary workaround for incompatibility of jmlr class with filename handeling in 2019 latex release
% See https://tex.stackexchange.com/questions/513300/unable-to-compile-with-includegraphics-using-jmlr-cls
% https://github.com/MIDL-Conference/MIDLLatexTemplate/issues/3
% https://ctan.org/ctan-ann/id/mailman.3352.1579371965.2586.ctan-ann@ctan.org
\ClassWarning{midl}{Using a pre 1.25 version of jmlr.cls with an midl-specific work around}
\def\set@curr@file#1{\def\@curr@file{#1}}
}

% (Temporary?) workaround for counting float-only last pages
% See https://github.com/MIDL-Conference/MIDLLatexTemplate/issues/10
\usepackage{placeins}
\let\old@jmlrenddoc\@jmlrenddoc
\renewcommand*\@jmlrenddoc{\FloatBarrier\old@jmlrenddoc}

\jmlrproceedings{PMLR}{Proceedings of Knowledge Discovery Course - UNICT }

\renewcommand\cite{\citep}

\newcommand{\midljointauthortext}[1]{\nametag{\thanks{#1}}}
\newcommand{\midlotherjointauthor}{\nametag{\addtocounter{footnote}{-1}\footnotemark}}

\ifanonsubmission
 \newcommand{\midlauthor}[1]{\author{\Name[Anonymous]{Author name(s) withheld} \Email{email(s) withheld} \\ \addr Address withheld}}
 \newcommand{\midlacknowledgments}[1]{\acks{Acknowledgments withheld.}}
\else
 \newcommand{\midlauthor}[1]{\author{\begin{NoHyper}#1\end{NoHyper}}}
 \newcommand{\midlacknowledgments}[1]{\acks{#1}}
\fi

\newcommand*{\@midl@check@packages}{%
 \@ifpackageloaded{subcaption}{%
  \ClassError{midl}{Package 'subcaption' detected.\MessageBreak
  This package conflicts with the midl/jmlr class. \MessageBreak
  Please use the jmlr \string\subfigure\space command. \MessageBreak
  See http://mirrors.ctan.org/macros/latex/contrib/jmlr/jmlr.pdf%
  }{}}{}%

 \@ifpackageloaded{cleveref}{%
  \ClassError{midl}{Package 'cleverref' detected.\MessageBreak
  This package conflicts with the midl/jmlr class. \MessageBreak
  Please use the existing jmlr \string\xyzref\space commands. \MessageBreak
  See http://mirrors.ctan.org/macros/latex/contrib/jmlr/jmlr.pdf%
  }{}}{}%

 \@ifpackageloaded{acro}{%
  \ClassError{midl}{Package 'acro' detected.\MessageBreak
  This package conflicts with the midl/jmlr class. \MessageBreak
  Please use the acronym package instead%
  }{}}{}%

 \@ifpackageloaded{glossaries}{%
  \ClassError{midl}{Package ' glossaries' detected.\MessageBreak
  This package conflicts with the midl class. \MessageBreak
  Please use the acronym package instead%
  }{}}{}%
}

\AtBeginDocument{%
 \@midl@check@packages
 \let\@midl@check@packages\relax
}
