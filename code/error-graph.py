'''

+===========================================================+
|  _____ _             _   ____            _           _    |
| |  ___(_)_ __   __ _| | |  _ \ _ __ ___ (_) ___  ___| |_  |
| | |_  | | '_ \ / _` | | | |_) | '__/ _ \| |/ _ \/ __| __| |
| |  _| | | | | | (_| | | |  __/| | | (_) | |  __/ (__| |_  |
| |_|   |_|_| |_|\__,_|_| |_|   |_|  \___// |\___|\___|\__| |
|                                       |__/                |
+===========================================================+

+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+
| A PYTHON PROGRAM FOR MAKING GRAPHS |
+~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

+--------------------------+
| Liam Mulhall             |
| Instructor: Shayon Gupta |
| TA: Alexander Curtiss    |
| Data Structures          |
| Fall 2018                |
+--------------------------+

+-----------------------------------------------+
| This code is adapted from Sentdex's tutorial. |
|                                               |
| Link to Tutorial:                             |
| https://youtu.be/QyhqzaMiFxk                  |
+-----------------------------------------------+

+-----------------------------------------------+
| This program is called from my neural network |
| program. It generates and saves a graph to   |
| the primary working directory.                |
+-----------------------------------------------+

'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

##################################################
## "Error-Graph"
##################################################

style.use('ggplot')

x, y = np.loadtxt('Error.csv', delimiter = ',', unpack = True)

plt.plot(x, y)

plt.title('Recent Average Error Versus Training Round')
plt.xlabel('Training Round')
plt.ylabel('Recent Average Error')

plt.savefig('Error-Graph.png')
