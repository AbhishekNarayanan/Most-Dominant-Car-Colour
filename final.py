import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import webcolors
def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name





img = cv2.imread("op350.jpg")
img = cv2.resize(img,(600,400),interpolation=cv2.INTER_CUBIC)
img = img[150:300, 250:450]
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
clt = KMeans(n_clusters=3) #cluster number
clt.fit(img)

hist = find_histogram(clt)
#print(hist,"\n",clt.cluster_centers_)
colors=clt.cluster_centers_
percent=tuple(hist)
index=percent.index(max(hist))
col=colors[index]
#print(colors)
#print(percent)
#print(tuple(col))
#print(col)
bar = plot_colors2(hist, clt.cluster_centers_)

plt.axis("off")
plt.imshow(bar)
plt.show()
print(tuple(col))

lis=[0,0,0]
tupl=tuple(col)
tup1=int(tupl[0])
tup2=int(tupl[1])
tup3=int(tupl[2])
lis[0]=tup1
lis[1]=tup2
lis[2]=tup3
col=tuple(lis)

list = [(0,0,205), (205,92,92),(0,255,0),(189,183,107),(220,220,220),(0,0,0),(255,255,255)]
#tree = KDTree.construct_from_data(list)
#nearest = tree.query(col, t=1)



grey=["gainsboro","lightgrey","silver","darkgrey","grey","dimgrey","lightslategrey"]
blue=["darkslategrey","lightsteelblue","slategrey","indigo","aliceblue","mintcream","honeydew","azure","lightseagreen","bluevoilet","darkblue","navy","mediumblue","royalblue","darkslateblue","cadetblue","midnightblue","slateblue","rosybrown","steelblue","cornflowerblue","dodgerblue","deepskyblue","skyblue","lightblue","powderblue"]
red=["lightsalmon","salmon","lightcoral","sandybrown""indianred","crimson","firebrick","red","navojwhite","mistyrose""peachpuff","darkred","maroon","tomato","orangered","saddlebrown","chocolate","peru","burlywood","brown"]
white=["white","wheat","lavender","cornsilk","ghostwhite","ivory","snow"]
yellow=["gold","goldenrod","olive","yellow","khaki","darkkhaki","lightyellow","darkgoldenrod"]
green=["greenyellow","yellowgreen","lawngreen","olivedrab","green","green","lightgreen","springgreen"]


actual_name, closest_name = get_colour_name((col))

if(closest_name in grey):
    print("grey")
elif(closest_name in blue):
    print("blue")
elif(closest_name in red):
    print("red")
elif(closest_name in yellow):
    print("yellow")
elif(closest_name in green):
    print("green")
elif(closest_name in white):
    print("white")
else:
    dist=(min(list, key=lambda c: (c[0]- col[0])**2 + (c[1]-col[1])**2 +(c[2]-col[2]**2)))
    print("closest name:",closest_name)


    if(dist==list[0]):
        print("Blue")
    if(dist==list[1]):
        print("Red")
    if(dist==list[2]):
        print("Green")
    if(dist==list[3]):
        print("Yellow")
    if(dist==list[4]):
        print("grey")
    if(dist==list[5]):
        print("black")
    if(dist==list[6]):
        print("White")