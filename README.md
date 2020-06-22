# Sigmoid-SIRFH COVID Model

This model was created based on general [compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) by the [Asset Allocation team](https://conteudos.xpi.com.br/guia-de-investimentos/) of [XP Inc](https://www.xpinc.com/).

It was open sourced on the 19th of may 2020 with an [explanatory report](https://conteudos.xpi.com.br/guia-de-investimentos/relatorios/um-modelo-para-o-coronavirus/).

A quick glance and an introduction to the model is available on the [case study notebook](https://github.com/comunidadexp/sirfh-covid/blob/master/Case%20Study%20-%20Sigmoid%20SIRFH.ipynb) for South Korea, Italy, Spain, Germany, USA and Brazil.

# TL;DR

## Windows with virtual environment

Inside the folder `.\`, copy and paste on command line
```bat
python -m venv venv
.\venv\Scripts\activate
```

Then copy and paste
```bat
python -m pip install --upgrade pip
pip install --upgrade cython
pip install wheel
pip install -r requirements.txt
```

Then download Covid data that fits SIR classes requirements
```bat
git clone https://github.com/CSSEGISandData/COVID-19.git
```

The showroom is given by `Case Study - Sigmoid SIRFH.ipynb`. Give it a try =)

## Linux (Debian) with virtual environment

Inside the folder `./`, copy and paste on command line
```sh
python3.8 -m venv venv
source ./venv/bin/activate
```

Then copy and paste
```sh
python -m pip install --upgrade pip
pip install --upgrade cython
pip install wheel
pip install -r requirements.txt
```

Then download Covid data that fits SIR classes requirements
```bat
git clone https://github.com/CSSEGISandData/COVID-19.git
```

The showroom is given by `Case Study - Sigmoid SIRFH.ipynb`. Give it a try =)

# Requirements

All the libraries required are contained in the [Anaconda distribution](https://www.anaconda.com/).

The other requirements concern data sources. Most data comes from [Johns Hopkins University repository](https://github.com/CSSEGISandData/COVID-19). **The code needs a working copy of that repository**. The path to it should be passed to any `SIR` or `SIRFH` classes as the parameter `dir`. The default path is the root directory of this repository. Just clone the JHU repo into the same folder as this one. 

Additionally, we added two spreadsheets with total population data and quarantine declaration dates for the studied countries. Should other countries be added, they must be named exactly according to the JHU files standard.

# Overview

First-timers should check the [case study notebook](https://github.com/comunidadexp/sirfh-covid/blob/master/Case%20Study%20-%20Sigmoid%20SIRFH.ipynb) that contains several examples of the *Sigmoid-SIRFH* usage and the visualization tools.

The code is mostly implemented in the `SIR_models.py` file that consists of three classes.

The base class is the `SIR` class which is inherited by all others and implements a simple SIR model.

Secondly, the `SIRFH` class implemented the *SIRFH* model without time dynamics for `beta`.

Lastly, the `SIRFH_Sigmoid` class extends the `SIRFH` to add time dynamics to beta.

The main methods are 

* `load_data` which handles all the loading and cleaning of the data 
* `estimate` which estimates parameters for the specified country
* `predict` which uses the estimated parameters to generate compartment series solving the differential equations below
* `model` which implements the compartments dynamics

# Motivation

The model was mainly motivated by a few shortcoming of other approaches. Specifically, we believe the general SIR-like approach is jeopardized by the fact that:
* It depends on confirmed cases data, which likely suffers from significant under-reporting, thus being low-quality data.
* It depends on recovered cases data, which is visibly not timely reported. (for instance, in Brazil recovered cases jumped from 3k to 14k on a day)
* Most models do not split the *removed* compartment in deaths and recoveries.
* Most models do not consider the health infrastructure, which is highly significant to policy makers
* The `beta` parameter is usually constant, allowing for a slow finding of the lockdown beta.

Also, many econometric approaches are jeopardized by the fact that, by definition, they rely on other countries spread rhythm, even though countries had very different lockdown-like responses to covid, specially considering its timing.  

## Advantages

The proposed model has the following advantages:
* It splits the *infected* compartment into hospital and non hospital cases, allowing for a health infrastructure burden estimation
* It splits the *removed* compartment into *fatalities* and *recovered*, allowing for easily observed coefficients, such as: (i) number of days until the symptoms disappear; (ii) number of days until hospital cases recover; and (iii) number of days until deadly cases find its end
* It adds a time-dynamic to the *beta* coefficient, so that it fastly adapts to the lockdown regime change.

# Model

The model extends the [SIR model](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) by separating the *infected* compartment in *infected* (designating non-hospital cases) and *hospital* (designating hospital cases). The hospital compartment is internally divided into fatal-to-be cases and non-fatal, so that the recovery coefficients from the resulting compartment do not interfere with its sizes.

The compartments are modelled by differential equations which can be seen in the `model` method on the code.

The *susceptible* (S) compartment

<img src="https://render.githubusercontent.com/render/math?math=\frac{dS}{dt} = - \frac{\beta IS}{N}"> 
<br>

The *infected* (I) compartment

<img src="https://render.githubusercontent.com/render/math?math=\frac{dI}{dt} = (1 - \rho) \times \frac{\beta IS}{N} - \gamma_{I} I">
<br>

The hospitalized-to-recover compartment (Hr)

<img src="https://render.githubusercontent.com/render/math?math=\frac{dH_r}{dt} = \rho \times (1-\delta) \times \frac{\beta IS}{N} - \gamma_h H_r">
<br>

The hospitalized-to-be-fatal compartment (Hf)

<img src="https://render.githubusercontent.com/render/math?math=\frac{dH_f}{dt} = \rho \times \delta \times \frac{\beta IS}{N} - \omega H_f">
<br>

The recovered compartment (Hf)

<img src="https://render.githubusercontent.com/render/math?math=\frac{dR}{dt} = \gamma_{I} I_n + \gamma_h H_r">
<br>

The fatalities compartment (F)

<img src="https://render.githubusercontent.com/render/math?math=\frac{dF}{dt} =\omega H_f">
<br>

Finally, the time dynamics added to the `beta` parameter are done in a *[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)* fashion, in which the inflection point is set to be on the 7th day after the quarantine declaration date.