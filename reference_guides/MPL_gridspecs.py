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

#%%

fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
# fig = plt.figure()
# fig.suptitle("Test {}".format('TEst'), fontsize=20)
fig.suptitle("Test {}".format('TEst'))

gs = plt.GridSpec(3, 3)
gs.update(left=0.05, right=1, top=0.90)

# gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0:2, 0:2])
ax4 = plt.subplot(gs[-1, 0:2])
# ax2 = plt.subplot(gs[1, :-1])
ax3 = plt.subplot(gs[0, 2])
ax3 = plt.subplot(gs[1, 2])
ax3 = plt.subplot(gs[2, 2])
# ax5 = plt.subplot(gs[-1, -2])
plt.tight_layout()
plt.show()


#%%
if 0:
    ax_image = plt.subplot2grid((3,3), (0,1), colspan=2, rowspan=2, fig=fig) # topleft
    ax_hist = plt.subplot2grid((3,3), (2,1), colspan=2, fig=fig)                       # bottom left
    ax_1 = plt.subplot2grid((3,3), (0,0), fig=fig)            # right
    ax_2 = plt.subplot2grid((3,3), (1,0), fig=fig)            # right
    ax_3 = plt.subplot2grid((3,3), (2,0), fig=fig)            # right
    # fig.subplots_adjust(top=0.85)
    fig.tight_layout()
