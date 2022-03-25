Code and data for the research article titled “Physics Informed Neural Networks for Control Oriented Thermal Modeling of Buildings”, authored by Gargya Gokhale, Bert Claessens and [Chris Develder](http://users.atlantis.ugent.be/cdvelder/).

## Abstract
This paper presents a data-driven modeling approach for developing control-oriented thermal models of buildings. These models are developed with the objective of reducing energy consumption costs while controlling the indoor temperature of the building within required comfort limits. To combine the interpretability of white/gray box physics models and the expressive power of neural networks, we propose a physics informed neural network approach for this modeling task. Along with measured data and building parameters, we encode the neural networks with the underlying physics that governs the thermal behavior of these buildings. Thus, realizing a model that is guided by physics, aids in modeling the temporal evolution of room temperature and power consumption as well as the hidden state, i.e., the temperature of building thermal mass for subsequent time steps. The main research contributions of this work are: (1) we propose two variants of physics informed neural network architectures for the task of control-oriented thermal modeling of buildings, (2) we show that training these architectures is data-efficient, requiring less training data compared to conventional, non-physics informed neural networks, and (3) we show that these architectures achieve more accurate predictions than conventional neural networks for longer prediction horizons. We test the prediction performance of the proposed architectures using simulated and real-word data to demonstrate (2) and (3) and show that the proposed physics informed neural network architectures can be used for this control-oriented modeling problem.

## Contents
This repository presents the codes for the proposed physics informed neural network variants along with the training and test data sets that were used. Two variants have been proposed in the research article: (1) [PhysNet](PhysNet.py), and (2) [PhysRegMLP](PhysRegMLP.py).
Additionally, this repository also contains [data](.\data\) from a simulated building environment that was used for evaluating the performance of these two variants. 

## Citation
    @article{GOKHALE2022118852,
    title = {Physics informed neural networks for control oriented thermal modeling of buildings},
    journal = {Applied Energy},
    volume = {314},
    pages = {118852},
    year = {2022},
    issn = {0306-2619},
    doi = {https://doi.org/10.1016/j.apenergy.2022.118852},
    url = {https://www.sciencedirect.com/science/article/pii/S0306261922002884},
    author = {Gargya Gokhale and Bert Claessens and Chris Develder},
    keywords = {Physics-informed neural networks, Control-oriented modeling, Thermal building models, Deep learning},
    }

## Acknowledgment
This research was performed at the [AI for Smart Grids Research Group](https://ugentai4sg.github.io/) at [IDLAB, UGent--imec](https://www.ugent.be/ea/idlab/en). Part of this research has received funding from the European Union's Horizon 2020 research and innovation programme for the projects [BRIGHT](https://www.brightproject.eu/), [RENergetic](https://www.renergetic.eu/) and [BIGG](https://www.bigg-project.eu/).

## Contact
If you have any questions, please contact me at [gargya.gokhale@ugent.be].
