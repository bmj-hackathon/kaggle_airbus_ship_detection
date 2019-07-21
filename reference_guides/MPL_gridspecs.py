import imutils
from imutils.mj_paper import PAPER
fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
import matplotlib.pyplot as plt
fig = plt.figure()
ax_image = plt.subplot2grid((3,3), (0,0), colspan=2, rowspan=2) # topleft
ax_hist = plt.subplot2grid((3,3), (0,2))            # right
ax_1 = plt.subplot2grid((3,3), (1,2))            # right
ax_2 = plt.subplot2grid((3,3), (2,2))            # right
ax_3 = plt.subplot2grid((3,3), (2,0), colspan=2)                       # bottom left
fig.tight_layout()

plt.show()

