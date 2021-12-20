"""
Author: Jan Swierczek-Jereczek
Date: 24.11.2021

Library for visualising yelmo results in interactive way.
"""

# %%
######################################################################
######################## Import packages #############################
######################################################################

# Standard libraries.
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as matanim

import re                                   # For string list sorting.
import os                                   # For file navigation.
import subprocess                           # Allow shell commands for crop.
from IPython import display                 # Important to allow dynamic display.
from ipywidgets import interact             # Interact within a jupyter notebook
from collections import defaultdict         # Allow nested dictionnaries.
from ppy_colormaps import load_color_map   # To load pre-defined color maps.

#%%
######################################################################
####################### Helper functions #############################
######################################################################

# Natural key for the python sort function --> sorts as a human would.
# Found on Stackoverflow <3
def atoi(text):
  return int(text) if text.isdigit() else text
def natural_keys(text):
  return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# Get the files from parent folder and sort them.
def get_file_lists(path):

  # Initilise.
  list_nc1D = []
  list_nc2D = []

  # Fill lists with paths of result files.
  for root, dirs, files in os.walk(path):
    for file in files:
      if(file.endswith("yelmo1D.nc")): list_nc1D.append(os.path.join(root,file))
      if(file.endswith("yelmo2D.nc")): list_nc2D.append(os.path.join(root,file))

  # Use natural keys to sort the file paths.
  list_nc1D.sort(key=natural_keys)
  list_nc2D.sort(key=natural_keys)

  return list_nc1D, list_nc2D

# Provide an "indices" vector to specify which experiments should be kept.
def filter_file_lists(list_nc1D, list_nc2D, indices):
  list_nc1D_trunc, list_nc2D_trunc = [], []

  for i in indices:
    list_nc1D_trunc += [list_nc1D[i]]
    list_nc2D_trunc += [list_nc2D[i]]

  return list_nc1D_trunc, list_nc2D_trunc

# Print the sorted list with attributed indices.
# Thus, user knows which index to slide to for corresponding file.
def print_indexed_namelist(list_nc1D, list_nc2D):
  i=0
  for name1D, name2D in zip(list_nc1D, list_nc2D):
    print("index:",i,"   ",name1D, name2D)
    i+=1
    print("-----------------------------")


def load_variables(instance, input_varlist):
  instance.varlist += input_varlist
  for path in instance.list_nc:
    with xr.open_dataset(path) as f:
      for var in input_varlist:
        instance.sim_full[path][var] = f[var].copy()
  f.close()

def remove_variables(instance, input_varlist):
  for var_name in input_varlist:
    for path in instance.list_nc:
      del instance.sim_full[path][var_name]
    instance.varlist.remove(var_name)

def inherit_plot_spec(instance, nrw, ncl, plot_size, hl_lw):
  instance.nrw = nrw
  instance.ncl = ncl
  instance.plot_size = plot_size
  instance.hl_lw = hl_lw

# Save plot under the specified name.
def save_plot(instance, file_name):
  instance.plot_file = file_name
  instance.fig.savefig(file_name, bbox_inches="tight")

# Run pdf crop as shell command.
def crop_plot(instance):
  subprocess.call("pdfcrop "+instance.plot_file, shell=True)

# Get the variable name, values, min, max and unit for a given index of a provided list.
# If frame and sim have length 2: we can load two results at the same time (for difference plots).
def get_var(var_list, i, sim, frame):
  var_name = var_list[i]
  if isinstance(frame, (list, tuple, np.ndarray)):
    var1, var2 = sim[0][var_name][frame[0],:,:], sim[1][var_name][frame[1],:,:]
    var_unit = var1.units
    var = var1 - var2
    var_min, var_max = np.min(var), np.max(var)
  else:
    var = sim[var_name][frame,:,:]
    var_unit = var.units
    var_min, var_max = np.min(var), np.max(var)
  return var_name, var, var_min, var_max, var_unit

# Handle indices:
def handle_index(instance, i):
  i1, i2 = int(i/instance.ncl), i%instance.ncl
  return i1, i2

# Return associated axis:
def handle_axis(instance, i1, i2):
  if instance.nrw==1 and instance.ncl==1: cur_ax = instance.axs
  elif instance.nrw==1: cur_ax = instance.axs[i2]
  elif instance.ncl==1: cur_ax = instance.axs[i1]
  else: cur_ax = instance.axs[i1,i2]
  return cur_ax

def get_mask(instance2D, mask_ix=[2,3,4,5], visualise=False):

  mask = np.zeros([instance2D.nx, instance2D.ny])       # Initialise an integer mask.
  for ix in mask_ix:
    mask += (instance2D.region_mask == ix).astype(int)  # Set desired regions to 1.
  mask = np.array(mask)
  if visualise:
    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    ax.imshow(mask)
    ax.invert_yaxis()
    plt.show(fig)
  return mask

def mask_2D_to_1D(instance1D, instance2D, var_list, mask, scale_factor=1, average_method="ice-sheet"):

  # Speed up by cutting everything outside the mask
  mask_rows = np.sum(mask, axis=0)
  mask_cols = np.sum(mask, axis=1)
  
  i1 = (mask_rows!=0).argmax(axis=0)
  i2 = -(np.flip(mask_rows)!=0).argmax(axis=0)
  j1 = (mask_cols!=0).argmax(axis=0)
  j2 = -(np.flip(mask_cols)!=0).argmax(axis=0)
  # print(mask_rows.shape, mask_cols.shape, i1, i2, j1, j2)
  mask = mask[i1:i2, j1:j2]

  # Loop over files and variables.
  for file1, file2 in zip(instance1D.list_nc, instance2D.list_nc):
    for var_name in var_list:
      with xr.open_dataset(file2) as f:
          var = f[var_name].copy()             # Open the 2D var.
          var = scale_factor*var[:, i1:i2, j1:j2]
          masked_var = np.multiply(var, mask)               # Only keep value at points with int_mask=0.
          if average_method=="ice-sheet": masked_var1D = masked_var.sum(axis=(1,2))/(masked_var>0).sum(axis=(1,2))
          elif average_method=="fixed-zone": masked_var1D = masked_var.mean(axis=(1,2))
          elif average_method==None: masked_var.sum(axis=(1,2))
          instance1D.sim_full[file1]["masked_"+var_name] = masked_var1D # Store spatial mean in 1D data object.
      f.close()


#%%
######################################################################
############### Class for Post-Processing 1D-data ####################
######################################################################

class postpro_data1D:
  
  # Init function of 1D-data post-processing class
  def __init__(self, list_nc1D):

    # Inherit the input.
    self.list_nc = list_nc1D
    self.varlist = []
    
    self.n_exp = len(list_nc1D)       # Number of result files within the folder.
    self.sim_full = defaultdict(dict) # Create empty (nested) dictionnary for storing simulation results.

  # Plot function for 1D outputs
  def plot1D(self, ix, plotvar1Dlist, use, plot_features=None):
    
    if use == "init": plt.ioff()
    elif use == "final": plt.ion()

    # Initialise the figure and only use latex interpreter if labels provided.
    self.fig, self.axs = plt.subplots(nrows=self.nrw, ncols=self.ncl, figsize=self.plot_size)
    self.handle = display.display(self.fig, display_id=True)
    self.plot1D_dict = defaultdict(dict)
    
    if plot_features["xlabels"] == None and plot_features["ylabels"] == None: plt.rcParams['text.usetex'] = False
    else: plt.rcParams['text.usetex'] = False

    for j in range(self.n_exp):
      for i in range(self.nrw*self.ncl):

        self.plotvar1Dlist =  plotvar1Dlist
        path, var_name = self.list_nc[j], plotvar1Dlist[i]
        i1, i2 = handle_index(self, i)
        cur_ax = handle_axis(self, i1, i2)
        x = self.sim_full[path][var_name]

        # Highlight the experiment specified in the input.
        if j==ix:
          self.plot1D_dict[var_name][str(j)], = x.plot.line(ax=cur_ax, color="tab:blue", zorder=2)
        else:
          self.plot1D_dict[var_name][str(j)], = x.plot.line(ax=cur_ax, color="tab:gray", zorder=1)

        # Only set labels when we arrived at the last experiment
        if path == self.list_nc[-1]:
          if plot_features["ylabels"] is not None: 
            cur_ax.set_ylabel(plot_features["ylabels"][i])
          if plot_features["xlabels"] is not None: 
            cur_ax.set_xlabel(plot_features["xlabels"][i])
          if plot_features["xticklabels"][i] is not None:
            cur_ax.set_xticks(plot_features["xticks"][i])
            cur_ax.set_xticklabels(plot_features["xticklabels"][i])
          cur_ax.grid(plot_features["grid_bool"])

    plt.tight_layout()
  
  def update1D(self, ix):
    
    ix = int(ix)    # For some widget types, ix is passed as a string --> make sure it's an int.
    plt.ion()

    for j in range(self.n_exp):
      for i in range(self.nrw*self.ncl):

        path, var_name = self.list_nc[j], self.plotvar1Dlist[i]
        x = self.sim_full[path][var_name]
        
        # Highlight the experiment specified in the input.
        self.plot1D_dict[var_name][str(j)].set_ydata(np.array(x))
        if j==ix:
          self.plot1D_dict[var_name][str(j)].set_color("tab:blue")
          self.plot1D_dict[var_name][str(j)].set_zorder(2)
          self.plot1D_dict[var_name][str(j)].set_linewidth(self.hl_lw)
        else: 
          self.plot1D_dict[var_name][str(j)].set_color("tab:gray")
          self.plot1D_dict[var_name][str(j)].set_zorder(1)
          self.plot1D_dict[var_name][str(j)].set_linewidth(1)
    self.handle.update(self.fig)

# Function to start an interactive 1D plot.
def interactive_1Dplot(instance1D, plotvar1Dlist, nrw, ncl, plot_size, hl_lw, plot_features, widget_type=None):
  
  inherit_plot_spec(instance1D, nrw, ncl, plot_size, hl_lw)

  if widget_type == None:
    instance1D.plot1D(0, plotvar1Dlist, use="final", plot_features=plot_features)

  else:
    instance1D.plot1D(0, plotvar1Dlist, use="init", plot_features=plot_features)

  if widget_type=="dropdown":
    # In order to have a dropdown menu, we need to have a list of strings.
    ix_list = map(str, np.arange(instance1D.n_exp))
    interact(instance1D.update1D, ix=ix_list)

  elif widget_type=="slider":
    interact(instance1D.update1D, ix=(0, instance1D.n_exp-1))


#%%
######################################################################
############### Class for Post-Processing 2D-data ####################
######################################################################

class postpro_data2D:

  # Init function of 2D-data post-processing class.
  def __init__(self, list_nc2D):
    
    # Inherit the input.
    self.list_nc = list_nc2D
    self.n_exp = len(self.list_nc)
    
    # Save all the data in a dictionnary.
    self.sim_full = defaultdict(dict)

    # For the 2D case, some variables have to be loaded.
    self.varlist = []
    obligatory_var = ["H_ice", "uxy_s", "H_ice_pd_err", "uxy_s_pd_err", "f_ice", "mask_bed"]
    load_variables(self, obligatory_var)

    mask_path = "data/ANT-32KM_REGIONS.nc"
    with xr.open_dataset(mask_path) as f:
      self.region_mask = f["mask_regions"].copy()
    f.close()

    self.feature_dict = defaultdict(dict) # Nested dictionnary saving color informations.
    for var_name in self.varlist:
      load_color_map(self, var_name)

    self.grline_precision = 1           # in ]0, 1], 1 = max precision in exctracted grline.
    self.extract_grline()               # Extract current-day grounding line for comparison.

    # Plotting constants
    self.N_cont = 10                  # number of plotted contour lines.
    self.cb_w, self.cb_h = 0.03, 0.45 # colorbar width and height.
    self.cb_x = [-self.cb_w, 1]       # x-coordinate of colorbar for column 0 and 1.
    self.cb_y = [0.52, 0.02]          # y-coordinate of colorbar for row 0 and 1.
  
  def delete_variable(self):
    # use to clean up the data instance if it is too large
    return

################ Functions for error computation #####################

  # Compute the error matrix for every file of the parent folder.
  def get_error(self,  error_weight):
    
    self.error_list = ["H_ice_pd_err", "uxy_s_pd_err", "grline_err"]
    self.error_weight = np.array(error_weight)
    self.n_metrics = len(error_weight)

    # Error_matrix: [index, wrmse, rmse_1, rmse_2, rmse_3].
    self.n_prefix_columns = 2                         # Columns 0 and 1 as "prefixes" for rmse_1, rmse_2, rmse_3.
    self.error_matrix = np.zeros([self.n_exp, self.n_prefix_columns + self.n_metrics])
    self.error_matrix[:, 0] = np.arange(self.n_exp)   # Set index column.
    

    j = 0
    for path in self.list_nc:
      self.sim = self.sim_full[path]
      mse_vec = np.zeros(self.n_metrics)

      for i in range(self.n_metrics):
        mse_vec[i] = self.get_single_mse(i)

      rmse_vec = np.round(np.sqrt(mse_vec), 3)
      self.error_matrix[j, self.n_prefix_columns:] = rmse_vec
      wrmse = self.get_weighted_rmse(rmse_vec)
      self.error_matrix[j, 1] = np.round(wrmse, 3)
      j += 1

  # Function to extract grounding line as a bool field stored in self.ref_grline.
  def extract_grline(self):
    # Get the mask and the space variables from reference observational data.
    
    path = "data/ANT-32KM_TOPO-BedMachine.nc"
    with xr.open_dataset(path) as f:
      self.mask = f["mask"].copy()
      self.X, self.Y = f["x2D"].copy(), f["y2D"].copy()
    self.nx, self.ny = self.X.shape

    # Obtain the grounding line by extracting it from matplotlib contour cmd.
    plt.ioff()                                        # Prevent from plotting.
    cs = plt.contour(self.mask==2, linewidths=.5, colors="k")
    plt.close()
    plt.ion()                                         # Turn plotting on again.
    self.ref_grline = np.full([self.nx, self.ny], False)        # Create False boolean field.
    n_paths = len(cs.collections[0].get_paths())      # Get number of contours.
    ix = int(n_paths*self.grline_precision)           # Truncate some of them.
    for p in cs.collections[0].get_paths()[:ix]:      # Iterate over truncated.
      v = p.vertices                                  # Each path has vertices.
      for vertex in v.astype(int):                    # Iterate over vertices.
        self.ref_grline[vertex[1], vertex[0]] = True  # Contour --> bool=True.

  # Computation of RMSE for 2D-field or grline.
  def get_single_mse(self, var_ix):
    var = self.error_list[var_ix]
    if var == "grline_err":
      mse = self.get_grline_mse()
    else:
      self.check_ice = np.array(self.sim["f_ice"][-1,:,:]) > 0
      var_sq = np.array(self.sim[var][-1, :, :])**2
      mse = 1/(self.check_ice.flatten().sum())*(var_sq[self.check_ice].flatten().sum())
    return mse

  # Computation of MSE for grline (1D space embedded in 2D one --> specific function).
  def get_grline_mse(self):
    index_grline = self.sim["mask_bed"][-1, :, :] == 4
    X,Y = self.X, self.Y
    X_grline, Y_grline = np.array(X)[index_grline].flatten(), np.array(Y)[index_grline].flatten()
    X_ref_grline, Y_ref_grline = np.array(X)[self.ref_grline].flatten(), np.array(Y)[self.ref_grline].flatten()

    l_cur = len(X_grline)
    l_ref = len(X_ref_grline)
    coor_ref_grline = np.hstack((X_ref_grline.reshape([l_ref, 1]), Y_ref_grline.reshape([l_ref, 1])))
    coor_grline = np.hstack((X_grline.reshape([l_cur, 1]), Y_grline.reshape([l_cur, 1])))

    se = 0
    for x in coor_grline:
      euclid = np.sum((x-coor_ref_grline)**2, axis=1)
      se += np.min(euclid)
    mse = 1/l_cur * se
    return mse

  # Weight the different RMSEs by factors leading to adimensionalisation and priorisation.
  def get_weighted_rmse(self, rmse_vec):
    return np.dot(rmse_vec, self.error_weight)

  def get_min_error_candidates(self, n_candidates):
    crit_vec = self.error_matrix[:, 1]
    sorted_indices = crit_vec.argsort()[:n_candidates]
    self.sorted_error = self.error_matrix[sorted_indices, :]

########################## Plot helpers ##############################

  def init_plot(self, plotvar2Dlist):
    plt.ioff()
    self.fig, self.axs = plt.subplots(nrows=2, ncols=2, figsize=self.plot_size, sharex=True, sharey=True)
    self.handle = display.display(self.fig, display_id=True)
    self.plot_dict = defaultdict(dict)
    self.var_list = plotvar2Dlist

  def get_ticklabels(self, tickstyle):
    if tickstyle == None:
      xticklabels, yticklabels = [], []
    elif tickstyle == "km":
      N_tick = 9      # number of ticks
      xmin, xmax = np.min(self.X), np.max(self.X)
      ymin, ymax = np.min(self.Y), np.max(self.Y)
      xticklabels = np.round(np.linspace(xmin, xmax, N_tick))
      yticklabels = np.round(np.linspace(ymin, ymax, N_tick))
    return xticklabels, yticklabels

  def plot_var_on_ax(self, var_name, var, ax, var_min, var_max):
    ax.contour(self.mask==2, linewidths=.5, colors="k")
    # Plot according to pre-defined colormaps if they exist.
    if var_name in self.feature_dict:
      self.plot_dict[var_name]["imshow"] = ax.imshow(var, norm=self.feature_dict[var_name]["norm"], cmap=self.feature_dict[var_name]["cmap"])
    else:
      self.plot_dict[var_name]["imshow"] = ax.imshow(var)
  
  def handle_coloraxis(self, var_name, i1, i2, var_units):
    # Handle the colorbar axis
    self.plot_dict[var_name]["coloraxis"] = self.fig.add_axes([self.cb_x[i2], self.cb_y[i1], self.cb_w, self.cb_h])
    self.plot_dict[var_name]["colorbar"] = self.fig.colorbar(self.plot_dict[var_name]["imshow"], cax=self.plot_dict[var_name]["coloraxis"], extend='max', orientation='vertical')
    if var_name in self.feature_dict: 
      self.plot_dict[var_name]["colorbar"].set_ticks(self.feature_dict[var_name]["levels"])
    self.plot_dict[var_name]["colorbar"].set_label(var_units)
    self.plot_dict[var_name]["colorbar"].ax.tick_params(labelsize=10,rotation=0)

  def handle_title(self, var_name, i1, i2):
    if "title" in self.feature_dict[var_name].keys():
      plt.rcParams['text.usetex'] = True
      self.axs[i1,i2].set_title(self.feature_dict[var_name]["title"])
    else:
      self.axs[i1,i2].set_title(var_name)

  def handle_ticks(self, var_name, i1, i2, xticklabels, yticklabels):
    if i1 == 1: 
      self.axs[i1,i2].set_xticklabels( xticklabels )
    if i2 == 0: 
      self.plot_dict[var_name]["coloraxis"].yaxis.set_ticks_position('left')
      self.plot_dict[var_name]["coloraxis"].yaxis.set_label_position('left')
      self.axs[i1, i2].set_yticklabels( yticklabels )
    else: 
      self.plot_dict[var_name]["coloraxis"].yaxis.set_ticks_position('right')
      self.plot_dict[var_name]["coloraxis"].yaxis.set_label_position('right')

  def plot_contour(self, var_name, var, i1, i2, mask, var_min, var_max):
    # if "grline" in self.plot_dict[var_name].keys(): self.plot_dict[var_name]["grline"].clear()
    self.plot_dict[var_name]["grline"] = self.axs[i1,i2].contour(mask==4, linewidths=.3, colors="r")
    contour_levels = np.linspace(var_min, var_max, self.N_cont)
    if "norm" in self.feature_dict[var_name].keys():
      self.plot_dict[var_name]["contour"] = self.axs[i1,i2].contour(var, contour_levels, linewidths=0.1, norm=self.feature_dict[var_name]["norm"], colors='k', linestyles='-')
    else:
      self.plot_dict[var_name]["contour"] = self.axs[i1,i2].contour(var, contour_levels, linewidths=0.1, colors='k', linestyles='-')

  def return2Dframe_number(self, ix):
    ix = int(ix)
    H_ice = self.sim_full[self.list_nc[ix]]["H_ice"]
    return H_ice.shape[0] - 1

######################## Plot Routines ###############################

  def plot2D(self, ix, frame, plotvar2Dlist, tickstyle):
    ix, frame = int(ix), int(frame)
    self.init_plot(plotvar2Dlist)
    xticklabels, yticklabels = self.get_ticklabels(tickstyle)
    sim = self.sim_full[self.list_nc[ix]]

    for i in range(self.nrw*self.ncl):
      # Handle indices and associated axies
      i1, i2 = handle_index(self, i)
      cur_ax = handle_axis(self, i1, i2)

      # Get values
      var_name, var, var_min, var_max, var_units = get_var(self.var_list, i, sim, frame)
      self.plot_var_on_ax(var_name, var, cur_ax, var_min, var_max)
      self.handle_coloraxis(var_name, i1, i2, var_units)
      self.axs[i1,i2].invert_yaxis()
      self.handle_title(var_name, i1, i2)
      self.handle_ticks(var_name, i1, i2, xticklabels, yticklabels)

    plt.tight_layout()
    plt.rcParams['text.usetex'] = False

  def update2D(self, ix, frame, with_contour=False, silent_update=False):
    
    if silent_update: plt.ioff()
    else: plt.ion()

    ix, frame = int(ix), int(frame)    # For some widget types, argument is passed as string --> make sure it's an integer.
    sim = self.sim_full[self.list_nc[ix]]
    mask = sim["mask_bed"][frame, :, :]

    for i in range(self.nrw*self.ncl):
      i1, i2 = handle_index(self, i)
      var_name, var, var_min, var_max, var_units = get_var(self.var_list, i, sim, frame)
      self.plot_dict[var_name]["imshow"].set_data(var)
      if with_contour: self.plot_contour(var_name, var, i1, i2, mask, var_min, var_max)

    if silent_update ==False: self.handle.update(self.fig)

  def plot2Ddiff(self, ix1, frame1, ix2, frame2, plotvar2Dlist, tickstyle=None):

    self.init_plot(plotvar2Dlist)
    ix1, ix2, frame1, frame2 = int(ix1), int(ix2), int(frame1), int(frame2)
    xticklabels, yticklabels = self.get_ticklabels(tickstyle)
    sim1, sim2 = self.sim_full[self.list_nc[ix1]], self.sim_full[self.list_nc[ix2]]

    for i in range(self.nrw*self.ncl):
      # Handle indices and associated axies
      i1, i2 = handle_index(self, i)
      cur_ax = handle_axis(self, i1, i2)

      var_name, var, var_min, var_max, var_units = get_var(self.var_list, i, (sim1, sim2), (frame1, frame2))
      var_name += "_diff"
      self.plot_var_on_ax(var_name, var, self.axs[i1, i2], var_min, var_max)
      self.handle_coloraxis(var_name, i1, i2, var_units)
      self.axs[i1,i2].invert_yaxis()
      self.handle_title(var_name, i1, i2)
      self.handle_ticks(var_name, i1, i2, xticklabels, yticklabels)

    plt.tight_layout()
    plt.rcParams['text.usetex'] = False

  def update2Ddiff(self, ix1, frame1, ix2, frame2, with_contour=False):
    
    ix1, ix2, frame1, frame2 = int(ix1), int(ix2), int(frame1), int(frame2)
    plt.ion()
    sim1, sim2 = self.sim_full[self.list_nc[ix1]], self.sim_full[self.list_nc[ix2]]
    mask1, mask2 = sim1["mask_bed"][frame1, :, :], sim2["mask_bed"][frame2, :, :]

    for i in range(self.nrw*self.ncl):
      i1, i2 = handle_index(self, i)
      var_name, var, var_min, var_max, var_units = get_var(self.var_list, i, [sim1, sim2], [frame1, frame2])
      var_name += "_diff"
      self.plot_dict[var_name]["imshow"].set_data(var)
      if with_contour:
        self.axs[i1, i2].contour(self.mask1==4, linewidths=.5, colors="tab:blue")
        self.axs[i1, i2].contour(self.mask2==4, linewidths=.5, colors="tab:red")

    self.handle.update(self.fig)

###################### 2D Time Visualisation #############################

  def evolution2Dplot(self, ix, frames, plotvar, nrw, ncl, plot_size):
    inherit_plot_spec(self, nrw, ncl, plot_size, hl_lw=None)
    fig, self.axs = plt.subplots(nrows=nrw, ncols=ncl, figsize=plot_size)
    sim = self.sim_full[self.list_nc[ix]][plotvar]
    for i in range(len(frames)):
      f = frames[i]
      i1, i2 = handle_index(self, i)
      cur_ax = handle_axis(self, i1, i2)
      cur_ax.imshow(sim[f,:,:])
      cur_ax.set_title(f"f = {f}")
    plt.tight_layout()
    plt.show(fig)

  def make2Dvideo(self, filepath, plotvar2Dlist, ix, fps=5, max_nf=None, tickstyle=None):
    print("Starting video generation...")
    PillowWriter = matanim.writers['pillow']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = PillowWriter(fps=fps, metadata=metadata)
    self.plot2D(ix, 0, plotvar2Dlist, tickstyle)

    if max_nf==None or max_nf>self.return2Dframe_number(ix): nf = self.return2Dframe_number(ix)
    else: nf=max_nf

    with writer.saving(self.fig, filepath+'.gif', dpi=100):
      for f in range(nf):
        if f%10==0: print("Getting frame number n°", f, "...")
        self.update2D(ix, f, silent_update=True)
        writer.grab_frame()


#%%
######################################################################
################### Class Interactive Plotting #######################
######################################################################
# For the 2D plots, the interactive features are handled by class-structures,
# as we need some nested interactive calls and therefore need inheritance.

# Class for interactive 2D-Plot.
class Interactive:

  def __init__(self, n_exp, plot_instance, plotvar2Dlist, widget_type):
    self.n_exp = n_exp
    self.plot_instance = plot_instance
    self.widget_type = widget_type
    self.plot_instance.plot2D(0, 0, plotvar2Dlist, tickstyle=None)

  def vary_frame(self, frame):
    self.plot_instance.update2D(self.ix, frame)

  def vary_source(self, ix):
    self.ix = ix
    nf = self.plot_instance.return2Dframe_number(ix)
    
    if self.widget_type=="dropdown":
      nf_list = map(str, np.arange(nf+1))
      interact(self.vary_frame, frame=nf_list)
    elif self.widget_type=="slider":
      interact(self.vary_frame, frame=(0,nf))


# Class for interactive 2D-Plot of difference maps.
class InteractiveDiff:

  def __init__(self, n_exp, plot_instance, plotvar2Dlist, widget_type="dropdown"):
    self.n_exp = n_exp
    self.plot_instance = plot_instance
    self.widget_type = widget_type
    self.plot_instance.plot2Ddiff(0, 0, 0, 100, plotvar2Dlist, tickstyle=None)
  
  def vary_frame(self, frame1, frame2):
    self.plot_instance.update2Ddiff(self.ix1, frame1, self.ix2, frame2)
  
  def vary_source(self, ix1, ix2):
    self.ix1, self.ix2 = ix1, ix2
    nf1, nf2 = self.plot_instance.return2Dframe_number(ix1), self.plot_instance.return2Dframe_number(ix2)

    if self.widget_type=="dropdown":
      nf1_list, nf2_list = map(str, np.arange(nf1+1)), map(str, np.arange(nf2+1))
      interact(self.vary_frame, frame1=nf1_list, frame2=nf2_list)
    elif self.widget_type=="slider":
      interact(self.vary_frame, frame1=(0,nf1), frame2=(0,nf2))

# Function operating the interactive call depending on the desired plot type.
def interactive_2Dplots(instance2D, plotvar2Dlist, nrw, ncl, plot_size, hl_lw, plot_type, widget_type):

  inherit_plot_spec(instance2D, nrw, ncl, plot_size, hl_lw)

  if plot_type == "simple":
    interactive_1 = Interactive(instance2D.n_exp, instance2D, plotvar2Dlist, widget_type)
    if widget_type=="dropdown":
      ix_list = map(str, np.arange(instance2D.n_exp))
      interact(interactive_1.vary_source, ix=ix_list)
    elif widget_type=="slider":
      interact(interactive_1.vary_source, ix=(0,instance2D.n_exp-1))
  elif plot_type == "diff":
    interactive_1 = InteractiveDiff(instance2D.n_exp, instance2D, plotvar2Dlist, widget_type)
    if widget_type=="dropdown":
      ix1_list, ix2_list = map(str, np.arange(instance2D.n_exp)), map(str, np.arange(instance2D.n_exp))
      interact(interactive_1.vary_source, ix1=ix1_list, ix2=ix2_list)
    elif widget_type=="slider":
      interact(interactive_1.vary_source, ix1=(0,instance2D.n_exp-1), ix2=(0,instance2D.n_exp-1))