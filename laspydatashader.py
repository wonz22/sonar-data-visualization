import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from laspy.file import File
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import datashader as ds
import datashader.transfer_functions as tf

currentPath= os.getcwd()
print(currentPath)
#os.chdir
#files= os.listdir(currentPath)
sample_data = File(r"C:\Users\heoji\Desktop\databasepython\datashader\original_data.las")
export_path = (r"C:\Users\heoji\Desktop\databasepython\datashader")

#inFile = File(sample_data, mode='r')
df = pd.DataFrame() 
df['X'] = sample_data.X 
df['Y'] = sample_data.Y 
df['Z'] = sample_data.Z
df['class'] = sample_data.classification
display(df)

cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(df, 'X', 'Y', ds.mean('Z'))
img = tf.shade(agg)#, cmap=['lightblue', 'darkblue'], how='log')
tf.set_background(tf.shade(agg, cmap=cm.inferno),"black")

# Create a dataframe containing only the lidar voxels 
class_df = df.loc[df['class'] == 6]

# Visualize with datashader
cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(class_df, 'X', 'Y', ds.mean('Z'))
img = tf.shade(agg)#, how='log')
tf.set_background(tf.shade(agg, cmap=cm.inferno),"black")

# Combine multiple classes of voxels that contain levels of vegetation(in datashader example..)
veg_df = df.loc[(df['class'] > 2) & (df['class'] < 6)]

# Visualize with datashader
cvs = ds.Canvas(plot_width=1000, plot_height=1000)
agg = cvs.points(veg_df, 'X', 'Y', ds.mean('Z'))
img = tf.shade(agg)#, how='log')
tf.set_background(tf.shade(agg, cmap=cm.inferno),"black")

# Use entire image containing other elements
X = df['X']
Y = df['Y']
Z = df['Z']

# Downsample x and y
ds_factor = 500
ds_x = X[::ds_factor] 
ds_y = Y[::ds_factor] 
ds_z = Z[::ds_factor] 

##### Export the gif
frames = []
identifier = 'bigO_tile_downsample_' + str(ds_factor) + 'x_surface_lidar'  

if not os.path.exists(export_path):
    os.makedirs(export_path)

fig = plt.figure(figsize = (10,10)) 
ax = fig.add_subplot(111, projection='3d')

# Plot the surface.
surf = ax.plot_trisurf(ds_x, ds_y, ds_z, cmap=cm.inferno,
                       linewidth=0, antialiased=False)

for angle in range(0, 360):
    ax.view_init(60, angle) # Higher angle than usually used of 30.
    ax.set_axis_off()

    # Draw the figure
    fig.canvas.draw()

    # Convert to numpy array, and append to list
    np_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    np_fig = np_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(np_fig)

imageio.mimsave(export_path + identifier + '.gif', frames)
