---
title: 'GeMMM: A Python package for sampling UK human movement patterns'
tags:
  - Python
  - population dynamics
  - epidemiology
  - human movement patterns
authors:
    - name: Jonathan Carruthers
      orcid: 0000-0002-1372-3254
      affiliation: '1'
    
    - name: Thomas Finnie
      orcid: 0000-0001-5962-4211
      affiliation: '1'

affiliations:
    - index: 1
      name: Analysis and Intelligence Assessment Directorate, Porton Down, UK Health Security Agency

date: 22 May 2025
bibliography: paper.bib
---

# Summary
Mathematical and computational models are frequently used to describe the spread of diseases transmitted by respiratory or close-contact routes [@legrand2007understanding;@coburn2009modeling;@garnett2002introduction]. From a single index case, many factors influence the speed and extent to which a disease spreads. One important consideration, reflected in the effectiveness of early contact tracing, is the areas to which an infected individual travels and the types of contact that are made once there [@hossain2022effectiveness]. While census data offers a snapshot of nationwide movement patterns, it lacks short-term variability. In contrast, travel surveys can provide greater detail but often suffer from low response rates across smaller geographical scales [@barbosa2018human]. More recently, the geolocation of users from major telecommunications providers has been used to construct mobility datasets with both high temporal and spatial resolution. Such datasets are valuable for developing more realistic models of disease spread, but may not be freely available to all researchers. Here we provide a mechanism to generate accurate, aggregate movement data at a fine spatial scale within the UK based on telecommunications data. This allows researchers and planners to quickly and easily work with such information without the complications inherent in working directly with the data.

# Statement of need
GeMMM (Generalized Mobile Movement Model) is a Python package that provides a simple approach for sampling realistic movement patterns across England, Scotland and Wales. These patterns are represented as origin-destination matrices, capturing the number of journeys between pairs of Middle Super Output Areas (MSOAs) at each hour of the day. GeMMM is desgined to be used alongside existing model frameworks to more accurately reflect the dynamics of disease transmission, but it can also support other applications, such as optimizing countermeasure delivery strategies based on expected travel behaviour.

The underlying probabilistic models are parameterised using mobility data collected by a major UK telecommunications provider over a three-month period at the beginning of 2023. By modelling variability in typical movement patterns, GeMMM enables the sampling of many unique realizations, in contrast to some mobility datasets that contain only a small number of repeated observations. In this way, users can propagate uncertainty in human mobility through their models to better understand the range of possible outcomes.

Currently, few Python packages provide the same functionality as GeMMM. For example, synthetic flows can be generated using standard migration models within scikit-mobility, however, these models must be parameterised by the user and therefore require external datasets [@pappalardo2022scikit]. Meanwhile, the individual based modelling framework, JUNE, incorporates mobility through commuting patterns derived from estimates by the Office for National Statistics and UK Department for Transport, but is restricted to a small subset of UK cities [@aylett2021june]. GeMMM is therefore well positioned to support future modelling efforts by providing access to realistic, simulated mobility data at a national scale.

# References
