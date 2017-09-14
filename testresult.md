
Example 1: sin^2(pi*x)*sin^2(pi*y)

* On uniform mesh with simple average
\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   441 &  1681 &  6561 & 25921\\\hline
$\| u - u_h\|$ &  0.01394 &  0.00344 &  0.00086 &  0.00021\\\hline
Order & -- &  2.02 &  2.   &  2.  \\\hline
$\|\nabla u - \nabla u_h\|$ &  0.39714 &  0.19032 &  0.09431 &  0.0471 \\\hline
Order & -- &  1.06 &  1.01 &  1.  \\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.04181 &  0.01051 &  0.00264 &  0.00066\\\hline
Order & -- &  1.99 &  1.99 &  2.  \\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  1.64494 &  0.80249 &  0.3989  &  0.19915\\\hline
Order & -- &  1.04 &  1.01 &  1.  \\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  0.31464 &  0.10367 &  0.03547 &  0.01234\\\hline
Order & -- &  1.6  &  1.55 &  1.52\\\hline
\end{tabular}
\end{table}

* On CVDT mesh with simple average

\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   499 &  1920 &  7566 & 29952\\\hline
$\| u - u_h\|$ &  0.01205 &  0.00296 &  0.00069 &  0.00017\\\hline
Order & -- &  2.03 &  2.1  &  2.03\\\hline
$\|\nabla u - \nabla u_h\|$ &  0.37264 &  0.17001 &  0.07189 &  0.03457\\\hline
Order & -- &  1.13 &  1.24 &  1.06\\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.03859 &  0.00981 &  0.00234 &  0.00059\\\hline
Order & -- &  1.98 &  2.07 &  1.99\\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  1.34357 &  0.65669 &  0.34152 &  0.17329\\\hline
Order & -- &  1.03 &  0.94 &  0.98\\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  0.20484 &  0.06668 &  0.02176 &  0.00814\\\hline
Order & -- &  1.62 &  1.62 &  1.42\\\hline
\end{tabular}
\end{table}

* On CVDT Mesh with harmonic average

\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   499 &  1920 &  7566 & 29952\\\hline
$\| u - u_h\|$ &  0.01216 &  0.00296 &  0.00069 &  0.00017\\\hline
Order & -- &  2.04 &  2.11 &  2.04\\\hline
$\|\nabla u - \nabla u_h\|$ &  0.38187 &  0.17156 &  0.07217 &  0.0346 \\\hline
Order & -- &  1.15 &  1.25 &  1.06\\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.03901 &  0.00987 &  0.00236 &  0.0006 \\\hline
Order & -- &  1.98 &  2.07 &  1.98\\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  1.35263 &  0.66216 &  0.344   &  0.17487\\\hline
Order & -- &  1.03 &  0.94 &  0.98\\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  0.2228  &  0.07085 &  0.02371 &  0.00914\\\hline
Order & -- &  1.65 &  1.58 &  1.37\\\hline
\end{tabular}
\end{table}


Example 2:  sin(2*pi*x)*sin(2*pi*y)

* On uniform mesh with simple average

\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   441 &  1681 &  6561 & 25921\\\hline
$\| u - u_h\|$ &  0.02908 &  0.00724 &  0.00182 &  0.00046\\\hline
Order & -- &  2.01 &  1.99 &  1.99\\\hline
$\|\nabla u - \nabla u_h\|$ &  0.93692 &  0.42536 &  0.20684 &  0.10261\\\hline
Order & -- &  1.14 &  1.04 &  1.01\\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.09441 &  0.02477 &  0.00643 &  0.00166\\\hline
Order & -- &  1.93 &  1.94 &  1.96\\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  4.88858 &  2.41195 &  1.21624 &  0.61711\\\hline
Order & -- &  1.02 &  0.99 &  0.98\\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  1.5065  &  0.51383 &  0.20083 &  0.08741\\\hline
Order & -- &  1.55 &  1.36 &  1.2 \\\hline
\end{tabular}
\end{table}

* On CVDT mesh with simple average

\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   499 &  1920 &  7566 & 29952\\\hline
$\| u - u_h\|$ &  0.0179  &  0.0045  &  0.00112 &  0.00028\\\hline
Order & -- &  1.99 &  2.01 &  2.  \\\hline
$\|\nabla u - \nabla u_h\|$ &  0.61027 &  0.27806 &  0.1316  &  0.06381\\\hline
Order & -- &  1.13 &  1.08 &  1.04\\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.05595 &  0.01519 &  0.00407 &  0.00113\\\hline
Order & -- &  1.88 &  1.9  &  1.85\\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  4.35481 &  2.14929 &  1.07906 &  0.54476\\\hline
Order & -- &  1.02 &  0.99 &  0.99\\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  1.14406 &  0.39178 &  0.15557 &  0.06365\\\hline
Order & -- &  1.55 &  1.33 &  1.29\\\hline
\end{tabular}
\end{table}

* On CVDT Mesh with harmonic average

\begin{table}[!htdp]
\begin{tabular}[c]{|c|c|c|c|c|}\\\hline
Dof &   499 &  1920 &  7566 & 29952\\\hline
$\| u - u_h\|$ &  0.01778 &  0.00449 &  0.00111 &  0.00028\\\hline
Order & -- &  1.98 &  2.01 &  2.01\\\hline
$\|\nabla u - \nabla u_h\|$ &  0.60852 &  0.27923 &  0.13189 &  0.06394\\\hline
Order & -- &  1.12 &  1.08 &  1.04\\\hline
$\|\nabla u - G(\nabla u_h)\|$ &  0.05602 &  0.01552 &  0.00421 &  0.00124\\\hline
Order & -- &  1.85 &  1.88 &  1.77\\\hline
$\|\Delta u - \nabla\cdot G(\nabla u_h)\|$ &  4.36727 &  2.14856 &  1.07652 &  0.54663\\\hline
Order & -- &  1.02 &  1.   &  0.98\\\hline
$\|\Delta u -  G(\nabla\cdot G(\nabla u_h))\|$ &  1.16629 &  0.3933  &  0.15138 &  0.06294\\\hline
Order & -- &  1.57 &  1.38 &  1.27\\\hline
\end{tabular}
\end{table}
