# Community

Parcels users and developers interact in a vibrant community on a few different platforms. Check out the cards below to see how you can interact with us.

`````{grid} 1 2 2 2
:gutter: 4
:padding: 2 2 0 0
:class-container: sd-text-center

````{grid-item-card} CLAM Community on Zulip
:img-top: https://raw.githubusercontent.com/CLAM-community/CLAM-community.github.io/ca1b93cb79410f7ffbbca7ae6860c5d5e0430d31/docs/assets/branding/svg/clam-full-white-buffer.svg
:shadow: md

If you are doing any kind of Lagrangian modelling and/or analysis check out the CLAM (Computational Lagrangian Analysis and Modelling) Community

```{image} https://img.shields.io/badge/Zulip-50ADFF?style=for-the-badge&logo=Zulip&logoColor=white
:width: 30%
:target: https://clam-community.zulipchat.com/login/
```

+++

```{button-link} https://clam-community.github.io/
:color: secondary
:expand:

To the CLAM website
```
````
````{grid-item-card} GitHub
:img-top: ../_static/github-logo.svg
:shadow: md

If you need more help with Parcels, try the Discussions page on **GitHub**. If you think you found a bug, please feel free to file an Issue.

+++

```{button-link} https://github.com/Parcels-code/parcels/discussions
:color: secondary
:expand:

Ask a question in the Discussions
```
```{button-link} https://github.com/Parcels-code/parcels/issues
:color: secondary
:expand:

Report a bug with an Issue
```
````
````{grid-item-card} Sharing user code
:shadow: md

Curious to see if someone has already written the custom `Kernel` you are thinking of or runs Parcels with the same hydrodynamic data? Check out the parcels_contributions repository and share examples with other users!

```{image} https://img.shields.io/badge/maintainer_needed-red
:width: 40%
```

+++

```{button-link} https://github.com/Parcels-code/parcels_contributions
:click-parent:
:color: secondary
:expand:

Share custom Parcels code
```
````
`````

(analysis-code)=

## Analysis code

The following is an alphabetically sorted list of tools for analysing Lagrangian trajectory output:

- [Lagrangian Diagnostics](https://lagrangian-diags.readthedocs.io/en/latest/) (![maintainer needed](https://img.shields.io/badge/maintainer_needed-red)): Are you interested in advanced analysis and diagnostics of Parcels output or Lagrangian trajectories in general? The Lagrangian Diagnostics project provides code and descriptions of different analyses.
- [Lagrangian Trajectories Toolbox](https://github.com/oj-tooth/lt_toolbox): A Python library dedicated to the post-processing, visualisation and analysis of Lagrangian particle trajectories. This library assumes trajectories are stored as tabular output (e..g, Parquet or CSV).
- [TrajAn](https://github.com/OpenDrift/trajan): A Python package for analysing and plotting ocean drifter and trajectory data stored, developed as part of the OpenDrift project. This library assumes trajectories are stored as CF-compliant Netcdf/Zarr output.

## Projects that use Parcels

The following is an alphabetically sorted list of projects that use Parcels:

- [LOCATE](https://github.com/UPC-LOCATE/LOCATE/): A collection of numerical tools developed within LOCATE ESA-funded project to build simulations of plastic particle dispersion in nearshore water.
- [PlasticParcels](https://github.com/Parcels-code/plasticparcels): A tool - based on Parcels - providing a modular and customisable collection of methods, notebooks, and tutorials for advecting virtual plastic particles with a wide range of physical properties.
- [pyPlume](https://github.com/jerukan/PyPlume): A collection of notebooks and methods made unifying the process of loading two-dimensional oceanic current vector fields from models and observations, simulating trajectory models, and analyzing and visualizing particle trajectories.
- [VirtualFleet](https://github.com/euroargodev/VirtualFleet): Make and analyse simulations of virtual Argo float trajectories
- [VirtualShip](https://virtualship.parcels-code.org/): A framework to plan and conduct a virtual research expedition, receiving measurements as if they were coming from actual oceanographic instruments.

## Other Lagrangian software

The following is an alphabetically sorted list of other Lagrangian ocean modelling and analysis software:

- [connectivity-modeling-system (CMS)](https://github.com/beatrixparis/connectivity-modeling-system)
- [Drifters.jl](https://github.com/JuliaClimate/Drifters.jl)
- [oceantracker](https://github.com/oceantracker/oceantracker)
- [OpenDrift](https://github.com/OpenDrift/opendrift)
- [TrackMPD](https://github.com/IJalonRojas/TrackMPD)
- [TRACMASS](https://www.tracmass.org/)
