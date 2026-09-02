# V2.6 sequential SLM feasibility review

## Decision

Two sequential phase-only masks are physically defensible for a qualified TessScope
in-silico experiment. They are not assumed to be instantaneous, lossless, or compatible
with every SLM. The claim boundary remains a hardware-ready simulation until measured on
a real detection-path device.

## Primary experimental evidence

- [Multiplexed phase-space imaging for 3D fluorescence microscopy](https://doi.org/10.1364/OE.25.014986)
  used an SLM in a microscope pupil plane to sequentially pattern coded apertures while
  capturing fluorescence images.
- [Self-contained and modular structured illumination microscope](https://pmc.ncbi.nlm.nih.gov/articles/PMC8367227/)
  advances an SLM pattern between camera exposures using the scientific camera's exposure
  trigger; its acquisition is explicitly a sequence of nine fluorescence images.
- [Super-resolution video microscopy of live cells by structured illumination](https://pmc.ncbi.nlm.nih.gov/articles/PMC2895555/)
  preloads patterns and synchronizes SLM changes, illumination, and camera transfer with
  hardware-timed triggers.
- [Spatial and spectral imaging of point-spread functions using an SLM](https://pmc.ncbi.nlm.nih.gov/articles/PMC6179356/)
  demonstrates fluorescence-emission PSF engineering with an SLM in a Fourier-plane
  detection path, matching TessScope's intended optical location.

## Official device evidence

- [Hamamatsu X15213-13 specifications](https://lcos-slm.hamamatsu.com/eu/en/lcos-slm/specific_wavelength_type/X15213-13.html)
  list a 60 Hz frame rate, 10 ms rise time, 25 ms fall time, and 97% light utilization at
  the stated 532 nm measurement condition.
- [Meadowlark high-speed 1024×1024 LCoS datasheet](https://www.meadowlark.com/wp-content/uploads/2022/04/SLM-High-Speed-1024x1024-Data-Sheet-2.pdf)
  lists automated sequencing, hardware triggers, internal frame memory, and approximately
  1 ms liquid-crystal response at 532 nm for its fastest configuration.

## Frozen qualifications

- The two exposures use a matched total expected-photon budget; a second mask does not
  create free photons.
- Switching overhead is device-dependent and must be added to camera exposure/readout in
  a hardware implementation.
- Polarization and diffraction losses can be material in fluorescence. Simulation uses
  matched throughput and cannot establish a real device's efficiency.
- Live motion, bleaching, SLM calibration, chromatic response, and synchronization are
  outside the current digital twin and remain hardware risks.
- A fair two-mask piecewise baseline receives the same number of masks, exposures,
  photons, observer, autofocus controller, and evaluation data.
