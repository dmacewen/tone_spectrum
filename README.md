# Exploration into Perception of Color and Metamerism

## WIP - commiting to test how Github renders markdown

## Tone Overview
Tone is a project that aims to address the challenge of getting an accurate foundation makeup match. The current best method for getting accurately matching foundation makeup is to go to a brick and mortar store and try on different shades. Making matters worse, skin tone and skin needs change throughout the year for many people which means the user will often need to go back to the store if their skin changes. 

Tone works to address the core issues by measuring the users skin tone with a mobile app and match them to the best foundation makeup for their needs.

## Tone Projects
|Repo | |
|---|---|
| [Tone Color Match](https://github.com/dmacewen/tone_colorMatch) | Primary image processing pipeline. Takes a set of images of the users face (taken by the iOS app) and records the processed colors to the database |
| [Tone iOS App](https://github.com/dmacewen/tone_ios) | Guides the user through capturing the images, preprocesses them, and sends them to the server |
| [Tone Server](https://github.com/dmacewen/tone_server) | RESTful API handling user data, authentication, and beta testing information. Receives images from the iOS app and starts color measuring jobs |
| [Tone Spectrum (This Repo)](https://github.com/dmacewen/tone_spectrum) | A deep dive into [metamerism](https://en.wikipedia.org/wiki/Metamerism_(color)) as a potential source of error for Tone in its current form |
| [Tone Database](https://github.com/dmacewen/tone_database) | SQL |
| [Tone SQS](https://github.com/dmacewen/tone_sqs) | Command line utility for sending SQS messages to the Color Match worker. Good for running updated Color Match versions on old captures |

## Tone Spectrum Writeup

### Overview

While beta testing Tone, I was noticing that I was not able to achieve the accuracy I was hoping to see. There are a [number of reasons](https://github.com/dmacewen/tone_colorMatch#tone-post-mortem) that could be causing this but one seemed particulary interesting: [Metamerism](https://en.wikipedia.org/wiki/Metamerism_(color)).

Wikipedia explains metamerism well: metamerism is a perceived matching of colors with different (nonmatching) spectral power distributions. This makes it hard or impossible to compare two different colors in some lighting conditions. This is actually a concept that I suspect shows up in the broader makeup industry. Many makeup users report their fustration of having foundation makeup match in the store, but not match when outside. Artifical light sources often do no emit a uniform spectrum of light. In a similar vein, phone screens also do not emit a uniform spectrum of light.

![wikipedia_spectral_power_distriputions](/svg/blood_spectrum.png)
