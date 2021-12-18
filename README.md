# Introduction

Yelmo is an ice sheet simulation framework developed at the *Universidad Complutense de Madrid*. It saves most of its results in .nc-files, which are commonly visualised in *ncview*, an interactive plotting GUI.

Even though *ncview* shows a great versatility, it also presents some drawbacks:
1. If we want to compare multiple results, we need to open an *ncview*-instance for each file.
2. Even by doing so, comparing 1D outputs can be tedious as they are plotted in seperated windows.
3. 2D outputs are even harder to compare.
4. The plots can only be modified up to a certain degree (e.g. colorscale, labels...) and cannot be saved into a vector-graphic format.
5. It is not possible to apply a mask to 2D variables in order to generate 1D variables.
6. Though a video of the 2D fields can be played, it cannot be directly saved into a publishable format.

Postprocess-yelmo (PPY) aims to offer these possibilities by a series of Python routines and classes. The use of postprocess-yelmo within a jupyter notebook allows us to interact with the plots and apply any modification if desired. Though all the above-mentioned drawbacks of *ncview* are solved by the use of PPY, it should be mentioned that *ncview* still offers a more developed GUI which is still irreplaceable in some cases.

# Features

As the features of PPY are best understood on a practical example, a jupyter notebook (*ppy_demo.ipynb*) demonstrates all its basic functionalities.

# Dependencies

PPY relies on standard python libraries. A .yml-environment is provided in the git repostory for any user to be able to run it rapidly.

# Before Getting Started

PPY relies on some .nc files, typical of the studied domain. For instance, in the case of Antarctica, the present-day state as well as a mask of the regions need to be provided. The very first step before getting started is to make sure you have the required files and have set the paths to it correctly (illustrated in *ppy_demo.ipynb*)
