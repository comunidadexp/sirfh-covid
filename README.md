# Sigmoid-SIRFH COVID Model

This model was created based on general [compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology) by the Asset [Allocation team](https://conteudos.xpi.com.br/guia-de-investimentos/) of [XP Inc](https://www.xpinc.com/).

It was open sourced on the 19th of may 2020 with an [explanatory report](https://conteudos.xpi.com.br/guia-de-investimentos/relatorios/um-modelo-para-o-coronavirus/).

A quick glance and introduction to the model is available on the [case study notebook](https://github.com/comunidadexp/sirfh-covid/blob/master/Case%20Study%20-%20Sigmoid%20SIRFH.ipynb) for South Korea, Italy, Spain, Germany, USA and Brazil.

# Requirements

#TODO COMPLETAR

# Motivation

The model was mainly motivated by a few shortcoming of other approaches. Specifically, we believe the general SIR-like approach is jeopardized by the fact that:
* It depends on confirmed cases data, which likely suffers from significant under-reporting, thus being low-quality data.
* It depends on recovered cases data, which is visibly not timely reported. (for instance, in Brazil recovered cases jumped from 3k to 14k on a day)
* Most models do not split the *removed* compartment in deaths and recoveries.
* Most models do not consider the health infrastructure, which is highly significant to policy makers
* $The `beta` parameter is usually constant, allowing for a slow finding of the lockdown beta.

Also, many econometric approaches are jeopardized by the fact that, by definition, they rely on other countries spread rhythm, even though countries had very different lockdown-like responses to covid, specially considering its timing.  

## Advantages

The proposed model has the following advantages:
* It splits the *infected* compartment into hospital and non hospital cases, allowing for a health infrastructure burden estimation
* It splits the *removed* compartment into *fatalities* and *recovered*, allowing for easily observed coefficients, such as: (i) number of days until the symptoms disappear; (ii) number of days until hospital cases recover; and (iii) number of days until deadly cases find its end
* It adds a time-dynamic to the *beta* coefficient, so that it fastly adapts to the lockdown regime change.

# Model
<img src="https://render.githubusercontent.com./render/math?math=\beta_test">
