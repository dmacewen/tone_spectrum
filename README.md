# Exploration into Perception of Color and Metamerism

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

Example of spectral distrubitions of different light sources from [wikipedia](https://upload.wikimedia.org/wikipedia/commons/b/b0/Spectral_Power_Distributions.png):

<p align="center">
    <img src="/readme_resouces/Spectral_Power_Distributions.png">
</p>

Parital light distribution of device screens is a potential source of error for Tone in two ways:

1. Using the same device, a user with changing skin tone may get the same results
2. Two different devices with different screens may report two skin tones as the same when they should differ, or vice versa

In order to analyze the spectral power distributions of light sources, I need a spectrometer.

### Building a Spectrometer

Measuring the spectral power distribution requires a spectrometer, but the types that you can read the output from are expensive, so I built my own.

Starting with a [basic spectroscope](https://www.amazon.com/EISCO-Resolution-Quantitative-Spectroscope-400-700/dp/B00FGARIAO) and attaching a cellphone allows me to capture images of the spectral distribution.

NOTE: This whole process is not incredibly accurate, but it should give an indication into whether metemerism could be an influencing factor

A few modifications were required to improve the quality of the images captured:
* Line the spectroscope with felt to reduce internal reflections
* Add a red filter over the number line to reduce the brightness of the numbers and reduce any diffraction from the numbers

The images are captured in a raw format to prevent any changes being performed before we start processing. 

Unprocessed output from the spectroscope:
<p align="center">
    <img src="/readme_resouces/sunlightRaw.jpg">
</p>

### Processing the Spectroscope Capture
These steps are applied in `imageToSpectrum.py`

1. The raw image is loaded
2. A crop is automatically applied around the numberline and the light spectrum
    * This is done by finding the number line using thresholding in the red color channel
    * Once the numbers are found, we know the spectrum is directly beneath
3. The color channels are separated out from each other, example shown below
    * The gradients from top to bottom are Red, Green, Blue

<p align="center">
    <img src="/readme_resouces/SunRGBSpectralDistributionImage.png">
</p>


4. In the spectrum, each color channel is converted into a set of (x, y) points, with y representing the magnitude of the channel brightness at that location
    * Each column of pixels in the spectrum sample can be averged to get a sampling for that location 
5. We can scale the Y-Axis between 0 and 1, and the X-Axis between the starting wavelength and the ending wavelength (taken from numberline)
    * The results, plotted, look like:
    * Important Note: This is uncalibrated, raw sensor data plotted. It does not tell us all that much yet

<p align="center">
    <img src="/readme_resouces/SunRGBSpectralDistribution_uncalibrated.png">
</p>

6. The curve is then written to a CSV file along with some metadata to be processed by virtualCamera






### Simulating Different Lighting Scenarios
