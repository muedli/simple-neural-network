+===========================================================+
|  _____ _             _   ____            _           _    |
| |  ___(_)_ __   __ _| | |  _ \ _ __ ___ (_) ___  ___| |_  |
| | |_  | | '_ \ / _` | | | |_) | '__/ _ \| |/ _ \/ __| __| |
| |  _| | | | | | (_| | | |  __/| | | (_) | |  __/ (__| |_  |
| |_|   |_|_| |_|\__,_|_| |_|   |_|  \___// |\___|\___|\__| |
|                                       |__/                |
+===========================================================+

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
| A SIMPLE ARTIFICIAL NEURAL NETWORK WRITTEN IN C++ |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

+--------------------------+
| Liam Mulhall             |
| Instructor: Shayon Gupta |
| TA: Alexander Curtiss    |
| Data Structures          |
| Fall 2018                |
+--------------------------+

+---------------+
| Documentation |
+---------------+

+-------------------------------------+
| ========= Notes to Reader ========= |
| - The code for the program is       |
|   extensively commented.            |
| - There is a link to a video        |
|   tutorial near the top of the code |
|   for the program.                  |
+-------------------------------------+

//////////////////////////////////////////////////
// CONTENTS
//////////////////////////////////////////////////

(1) Dependencies
(2) Instructions
(3) Neuron Class
(4) Network Class
(5) Main Function
(6) Miscellaneous Sections

//////////////////////////////////////////////////
// (1) Dependencies
//////////////////////////////////////////////////

+--------------------------------+
| ======== DEPENDENCIES ======== |
| (1) gxx 4.2.1                  |
| (2) Python 3.7.1               |
| (3) Matplotlib 3.0.2           |
| (4) NumPy 1.15.4               |
+--------------------------------+

//////////////////////////////////////////////////
// (2) Instructions
//////////////////////////////////////////////////

+--------------------------------+
| ======== INSTRUCTIONS ======== |
| (1) Download "error-graph.py"  |
|     and move it to the same    |
|     directory in which the     |
|     program resides.           |
| (2) Compile using g++.         |
| (3) Execute.                   |
|                                |
| - No command line arguments    |
|   required.                    |
| - No data files required.      |
+--------------------------------+

//////////////////////////////////////////////////
// (3) Neuron Class
//////////////////////////////////////////////////

The "Neuron" class contains all code related to
making neurons and all code related to
the mathematical operations performed at each
neuron.

//////////////////////////////////////////////////
// (4) Network Class
//////////////////////////////////////////////////

The "Network" class contains all code related to
building the network of neurons and all code
related to the mathematical operations which
yield error statistics.

//////////////////////////////////////////////////
// (5) Main Function
//////////////////////////////////////////////////

The main function contains all driver code. It
can be modified or scrapped and replaced.

It is possible to change the functionality of
the neural network. To do so, changes to the
driver code in the main function should be made.

//////////////////////////////////////////////////
// (6) Miscellaneous Sections
//////////////////////////////////////////////////

There are many miscellaneous sections of code.
Some of these sections are necessary for the
"bare bones" of the program. Some are only used
in the main function. It should be easy to tell
which sections are which.
