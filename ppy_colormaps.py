"""
A script defining color maps for some frequently plotted yelmo variables.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define colormaps created by Javier Blasco.
def load_color_map(instance, var_name):

  # Ice thickness
  if var_name == "H_ice":
    ncolors_H = 14
    H_min = 0.1
    H_max = 3500
    colors_H = plt.cm.Blues(np.linspace(0.8, 0.15, ncolors_H, endpoint=True))
    instance.feature_dict["H_ice"]["levels"] = np.concatenate([np.linspace(H_min, H_max, ncolors_H+1)])
    instance.feature_dict["H_ice"]["cmap"], instance.feature_dict["H_ice"]["norm"] = mcolors.from_levels_and_colors(instance.feature_dict["H_ice"]["levels"], colors_H)
    instance.feature_dict["H_ice"]["cmap"].set_over(plt.cm.Blues(20))
    instance.feature_dict["H_ice"]["title"] = r"$H_\mathrm{ice}$"

    # Ice shelf thickness
    ncolors_H_f = 5
    H_f_min = 0.01
    H_f_max = 500
    colors_H_f = plt.cm.autumn(np.linspace(0.95,0.15,ncolors_H_f,endpoint=True))
    instance.levels_H_f = np.concatenate([np.linspace(H_f_min,H_f_max,ncolors_H_f+1)])
    instance.cmap_H_f, instance.norm_H_f = mcolors.from_levels_and_colors(instance.levels_H_f, colors_H_f)
    instance.cmap_H_f.set_over(plt.cm.autumn(20))

  # Velocity
  elif var_name == "uxy_s":
    ncolors_u=3
    colors =['#ffffff','#d9c8e0','#bcb3d4','#9087ba','#61409b','#00b1ff','#ffef37','#e0d018','#ffa200','#ff0000']
    instance.feature_dict["uxy_s"]["levels"] = np.concatenate([[0,2,5,10,20,50,100,200,500,1000,2000]])
    instance.feature_dict["uxy_s"]["cmap"], instance.feature_dict["uxy_s"]["norm"] = mcolors.from_levels_and_colors(instance.feature_dict["uxy_s"]["levels"], colors)
    instance.feature_dict["uxy_s"]["cmap"].set_over(plt.cm.jet(240))
    instance.feature_dict["uxy_s"]["title"] = r"$u_\mathrm{s}$"

  # Ice thickness error
  elif var_name == "H_ice_pd_err":
    ncolors_H_err = 9
    min_H_err = -1350
    max_H_err = 1350
    colors_H_err = plt.cm.RdBu(np.linspace(0.2,0.8,ncolors_H_err,endpoint=True))
    instance.feature_dict["H_ice_pd_err"]["levels"] = np.concatenate([np.linspace(min_H_err,max_H_err,ncolors_H_err+1)])
    instance.feature_dict["H_ice_pd_err"]["cmap"], instance.feature_dict["H_ice_pd_err"]["norm"] = mcolors.from_levels_and_colors(instance.feature_dict["H_ice_pd_err"]["levels"], colors_H_err)
    instance.feature_dict["H_ice_pd_err"]["cmap"].set_over(plt.cm.RdBu(250))
    instance.feature_dict["H_ice_pd_err"]["cmap"].set_under(plt.cm.RdBu(0))
    instance.feature_dict["H_ice_pd_err"]["title"] = r"$\Delta H_\mathrm{ice}$"

  # Velocity error
  elif var_name == "uxy_s_pd_err":
    instance.feature_dict["uxy_s_pd_err"]["levels"] = np.concatenate([[-375,-175,-75,-37.5,-17.5,-5,5,17.5,37.5,75,175,375]])
    colors_u_err = plt.cm.seismic(np.linspace(0.15,0.85,instance.feature_dict["uxy_s_pd_err"]["levels"].size-1,endpoint=True))
    instance.feature_dict["uxy_s_pd_err"]["cmap"], instance.feature_dict["uxy_s_pd_err"]["norm"] = mcolors.from_levels_and_colors(instance.feature_dict["uxy_s_pd_err"]["levels"], colors_u_err)
    instance.feature_dict["uxy_s_pd_err"]["cmap"].set_under(plt.cm.seismic(0))
    instance.feature_dict["uxy_s_pd_err"]["cmap"].set_over(plt.cm.seismic(255))
    instance.feature_dict["uxy_s_pd_err"]["title"] = r"$\Delta u_\mathrm{s}$"