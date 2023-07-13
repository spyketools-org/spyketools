<!--<h1 align="center">SpykeTools</h1>-->

<h2 align="center">SpykeTools: A high-level python toolbox for advanced large-scale neuronal data analyses</h2>

<p align="center"><img src="docs/logo_spyketools.png"  width="50%"/></p>

<!--<p align="center">Online article</p>-->

<p align="center">
  <a href="https://spyketools-org.github.io/spyketools/">
    <img alt="documentation" src="https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat" />
  </a>
</p>   
-->

---

## Key features

+ Use of free, open-source programming language (i.e., Python).
+ Designed to efficiently cope with high-dimensional neural data and can scale to large channel counts.
+ Neurodata Without Borders (NWB) format as Input.
+ Users can call high-level functions rather than scripting together many modules.
+ Focus on running parallel jobs to analyse spiking neural data.
+ Extensive functionality for extraction of high-dimensional neural patterns.


## Setup
=== "(Recomended) Conda (Linux/Windows/MacOS)"
	+ Create a conda environment:
	```sh
	conda env create --file environment.yml
	```
	+ Activate your conda environment:
	```sh
	conda activate SpykeTools
	```
=== "Debian/Ubuntu"
	Open a terminal and do:
	```sh
	sudo apt-get install allensdk=0.14.4 jupyter numba scikit-learn python=3.6.10 umap-learn
	```


---

_The work on SpykeTools was supported by the BMF (Bundesministerium fuer Bildung und Forschung), Computational Life Sciences, project BINDA (031L0167), Germany._

