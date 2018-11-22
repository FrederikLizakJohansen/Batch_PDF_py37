# TimeResolved_script

Coverting total scattering data into PDF.

The script consists of multiple part.

 * Functions.

 * Creating dictionary.

 * Defining variables in dictionary and constructing path for data.

 * Importing data.

   - Either as a txt or binary with hdf5.

 * Background subtracktion .

   - Manually or automatically (not perfect).

 * Calculating PDF and constructing output-files.

 * Saving data

   - Either a txt files or binary with hdf5.

   - Including header

 * Generating pictures.

# FUTURE IMPROVEMENTS

 * Prevent background subtraction for returning negative values.

   - Need some optimization.

 * Validation methods of the background subtraction.

 * Nyquist.

# Bugs

 * Auto single scaling not working.
