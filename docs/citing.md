## Citing SpykeTools


If you use SpykeTools in your research, please cite our publication on biorxiv:

```
@article{sotomayor2023spyketools,
  title={
    SpykeTools: A High-level Python Toolbox for Advanced 
    Large-scale Neuronal Data Analyses
  },
  author={Sotomayor-G{\'o}mez, Boris and Vinck, Martin},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

Depending on the used methods, please cite the following works: 

+ SpikeShip distance[^1]
+ ISI, SPIKE, and RI-SPIKE distances (PySpyke)[^2]
+ Victor-Purpura distance (Elephant toolkit)[^3]

<details> 
<summary> Sotomayor-Gomez et al. (2023) - SpikeShip: A method for fast, unsupervised discovery of high-dimensional neural spiking patterns </summary>
```
@article{sotomayor2020spikeship,
  title={SpikeShip: A method for fast, unsupervised discovery of high-dimensional neural spiking patterns},
  author={Sotomayor-G{\'o}mez, Boris and Battaglia, Francesco P and Vinck, Martin},
  journal={bioRxiv},
  pages={2020--06},
  year={2020},
  publisher={Cold Spring Harbor Laboratory}
}
```
</details>

<details> <summary> [2] Mulansky et al. (2020) - PySpike - A Python library for analyzing spike train synchrony </summary>
```
@article{mulansky2016pyspike,
  title={PySpike—A Python library for analyzing spike train synchrony},
  author={Mulansky, Mario and Kreuz, Thomas},
  journal={SoftwareX},
  volume={5},
  pages={183--189},
  year={2016},
  publisher={Elsevier}
}
```
</details>

<details>
<summary> [3] Denker M et al. (2018) Collaborative HPC-enabled workflows on the HBP Collaboratory using the Elephant framework</summary>
```
@conference{elephant18,
author = {Denker, M. and Yegenoglu, A. and Grün, S.},
booktitle = {Neuroinformatics 2018},
title = {{C}ollaborative {HPC}-enabled workflows on the {HBP} {C}ollaboratory using the {E}lephant framework},
pages = {P19},
year = {2018}
doi = {10.12751/incf.ni2018.0019},
url = {https://abstracts.g-node.org/conference/NI2018/abstracts#/uuid/023bec4e-0c35-4563-81ce-2c6fac282abd},
}
```
</details>
<!--**Core modules of software**
+ Boris Sotomayor-Gomez.-->

!!! info
	SpykeTools uses an adapted version of the spike train metrics ISI, SPIKE, and RI-SPIKE from [PySpike](http://mariomulansky.github.io/PySpike/) and Victor-Purpura distance from [Elephant](https://elephant.readthedocs.io/en/v0.9.0/reference/toctree/spike_train_dissimilarity/elephant.spike_train_dissimilarity.victor_purpura_distance.html?highlight=victor-purpura#elephant.spike_train_dissimilarity.victor_purpura_distance) for compatibility reasons. 

[^1]:
	Sotomayor-Gómez, B., Battaglia, F. P., & Vinck, M. (2020). SpikeShip: A method for fast, unsupervised discovery of high-dimensional neural spiking patterns. bioRxiv, 2020-06. [Article](https://www.biorxiv.org/content/10.1101/2020.06.03.131573v4.abstract).
[^2]:
	Mario Mulansky, Thomas Kreuz, PySpike - A Python library for analyzing spike train synchrony, Software X 5, 183 (2016). [Article](https://drive.google.com/file/d/1vJA5q4eFCd2ASKGN8ANaDNBfQVpWBPXd/view).
[^3]: 
	Denker, M., Yegenoglu, A., & Grün, S. (2018). Collaborative HPC-enabled workflows on the HBP Collaboratory using the Elephant framework (No. FZJ-2018-04998). Computational and Systems Neuroscience. [Article](https://juser.fz-juelich.de/record/851308).
