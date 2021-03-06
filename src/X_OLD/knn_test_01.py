#%% OLD <<<

#%% KNN
size = (100, 100)
size = (10, 10)

pixels = cv2.resize(img, size)


#%%
contour = imutils.get_contour(mask)
M = cv2.moments(contour)
center = (round(M['m10'] / M['m00']), round(M['m01'] / M['m00']))
logging.info("center : '{}'".format(center))

#%%

img2 = fit_draw_ellipse(img, contour)
plt.imshow(img2)
plt.show()
img3 = fit_draw_axes_lines(img, contour)
plt.imshow(img3)
plt.show()

#%%

canvas = np.zeros((mask.shape[0], mask.shape[1], 3))
logging.info("Canvas {}".format(canvas.shape))
img2 = cv2.circle(canvas, (round(M['m10'] / M['m00']), round(M['m01'] / M['m00'])), 5, (0, 100, 100), -1)
cnt = contours[0]
cv2.drawContours(img2, [cnt], -1, (0, 100, 100), 10)

plt.imshow(img2)
plt.show()

# box_coords = imutils.get_bbox_p(mask)

y1, y2, x1, x2 = imutils.get_bbox(mask)
cv2.moments()
pts = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

poly = plt.Polygon(pts, closed=True, fill=False)

ax.add_patch(poly)

plt.show()

#%%
# turn rle example into a list of ints

# Example: 1 code
fname = '000155de5.jpg'
this_record = df[df['ImageId'] == fname]

r = df.loc[df['ImageId'] == fname, 'EncodedPixels']
for rle_string in r:
    print(rle_string)
    rle = [int(i) for i in rle_string.split()]
    pairs = list(zip(rle[0:-1:2], rle[1::2]))

rle = [int(i) for i in this_record['EncodedPixels'].split()]
# turn list of ints into a list of (`start`, `length`) `pairs`

# First 3 pixels:
pairs[:3]



#%%
# Get 10 files
fnames = [zf.filename for zf in random.sample(img_zip.filelist, 10)]
this_file = random.choice(img_zip.filelist).filename

img = imutils.load_rgb_from_zip(img_zip, this_file)



plt.imshow(img)
plt.show()

#%%
def extract_color_histogram(image, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()




# def plot_hist(img, ax):
#     color = ('r', 'g', 'b')
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)
#         plt.xlim([0, 256])
#     plt.show()
#




#%%
# The image is stored as a np.ndarray
# Attributes"
# .shape - x, y, z
# .min 0
# .max 255
#
# img_original = imutils.open_rgb(image_path)

plt.figure("TITLE")
plt.imshow(img)
plt.show()

plot_hist(img)
# hist = extract_color_histogram(img_original)

#%%
size=(32, 32)
img2 = cv2.resize(img, size)
plt.imshow(img2)
plt.show()

# plot_hist(img)

#%%


fig = plt.figure(figsize=PAPER['A4_LANDSCAPE'], facecolor='white')
fig.suptitle("Test {}".format('TEst'), fontsize=20)



nrows = 2
nrowplots = nrows * 2
ncols = 3
height_ratios = [3,1] * nrows
width_ratios = [1] * ncols
major_rows = [(r, r+1) for r in np.arange(0,nrows * 2,2)]
n_imgs = nrows * ncols
random.seed(42)
fnames = [zf.filename for zf in random.sample(img_zip.filelist, n_imgs)]
imgs = [imutils.load_rgb_from_zip(img_zip, this_file) for this_file in fnames]

fig, axes = plt.subplots(nrowplots, ncols,
                         gridspec_kw={'width_ratios': width_ratios, 'height_ratios':height_ratios })

for icol in range(axes.shape[1]):
    for imain, ihist in major_rows:
        print("Main:",imain,col)
        print("Hist:",ihist,col)
        this_img = imgs.pop(0)

        ax_main = axes[imain, icol]
        print(ax_main)
        ax_main.imshow(this_img)
        ax_main.get_xaxis().set_visible(False)
        ax_main.get_yaxis().set_visible(False)

        ax_hist = axes[ihist, icol]
        plot_hist(this_img, ax_hist)



plt.show()

#%%

for i, img_path in enumerate(sel_img_paths):
    logging.info("{}".format(img_path))
    ax_main = fig.add_subplot(ROWS, COLS, i + 1)
    img = mpl.image.imread(img_path)
    ax_main.imshow(img)
    ax_main.axis('off')
    # plt.title(str_label)
plt.show()

#%%
img2.flatten()

#%%

t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax_main = plt.subplots()
ax_main.plot(t, s)

ax_main.set(xlabel='time (s)', ylabel='voltage (mV)',
            title='About as simple as it gets, folks')
ax_main.grid()

fig.savefig("test.png")
plt.show()