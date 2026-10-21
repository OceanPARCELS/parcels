# 📖 Parcels v4 development

Supported by funding from the [WarmWorld](https://www.warmworld.de) [ELPHE](https://www.geomar.de/en/research/ongoing-projects/project-details/prj/376612?cHash=449a07d057ea0d890287bd2b8ca26e8b) project and an [NWO Vici project](https://www.nwo.nl/en/researchprogrammes/nwo-talent-programme/projects-vici/vici-2022), the Parcels team is working on a major update to the Parcels codebase.

The key goals of this update are

1. to support `Fields` on unstructured grids;
2. to allow for user-defined interpolation methods (somewhat similar to user-defined kernels);
3. to make the codebase more modular, easier to extend, and more maintainable;
4. to align Parcels more with other tools in the [Pangeo ecosystem](https://www.pangeo.io/#ecosystem), particularly by leveraging `xarray` more; and
5. to improve the performance of Parcels.

The timeline for the release of Parcels v4 is not yet fixed, but we are aiming for an early release in September 2026. This v4.0 will have support for unstructured grids and user-defined interpolation methods, but is not yet performance-optimised. It is meant to gather user feedback and should not be considered stable.

The pages below provide further background on the development of Parcels v4. You can think of this page as a "living" document as we work towards the release of v4.

```{toctree}
nojit
Parcels v4 Issue Tracker <https://github.com/Parcels-code/Parcels/issues?q=sort%3Aupdated-desc%20is%3Aissue%20is%3Aopen%20(milestone%3A%22Parcels%20v4.0%20(preview)%22%20OR%20milestone%3A%22Parcels%20v4.x%20(stable)%22)>
Parcels v4 migration guide <../user_guide/v4-migration>
```
