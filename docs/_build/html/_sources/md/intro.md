# Introduction

The main functionality of GeMMM is to generate origin-destination matrices that describe the number of journeys occurring between middle super output areas (MSOAs) in England, Scotland and Wales. These matrices are representative of key human movement patterns present in mobile telecoms data and can be used to construct more realistic mathematical and computational models of disease. GeMMM uses a probabilistic model that combines a Fourier series model and radiation model to sample the number of journeys between given pairs of MSOAs. The following examples outline two scenarios that can benefit from these sampled origin-destination matrices.

<font size='4'> **Epidemiological models** </font> <br>
The SIR model offers a simple mathematical description for the spread of an infectious disease throughout a population. *Susceptible* individuals become *Infected* through contact with other infected individuals, before entering a *Removed* state that encompasses distinct disease outcomes such as infection acquired immunity and infection induced death. One underlying assumption of this model is that the population is homogeneous and well mixed, that is, an infected individual has equal probability of infecting any of the remaining susceptible individuals{cite}`keeling2005networks`. This assumption quickly becomes invalid as the spatial scale over which we are modelling increases.

Metapopulation and individual based models offer a more realistic approach, with the population instead divided into sub-populations, or patches, each of which has independent epidemiological dynamics{cite}`keeling2008modeling`. Within a patch, the assumption of a well-mixed population still stands, but transmission between two patches can only arise due to the movement of individuals. Transmission is therefore more likely to occur between individuals in neighbouring inner-city locations, than individuals in poorly connected areas at opposite ends of the country.

<font size='4'> **Airborne release** </font> <br>
The aerosolization of a biological agent and its subsequent dispersion in the atmosphere can result in the infection of individuals across a large area. Examples of this include the release of *Legionella* bacteria from contaminated water sources within cooling towers and air-conditioning systems{cite}`dyke2019dispersion`, and bioterrorist attacks{cite}`green2019confronting`. Plume models can be used to describe how the concentration of bacteria in the air changes in space and time, whilst disease models inform us of the likelihood of infection{cite}`egan2015review`. Combining these with knowledge about the population under the plume around the time of release allows us to estimate the number of casualties. However, when it comes to planning the distribution of medical countermeasures, understanding the probable desinations of those individuals passing through the exposed area is equally important.


 

