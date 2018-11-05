# TimeResolved_script

Coverting total scattering data into PDF.

The script consists of multiple part.

 * Functions

 * Creating dictionary

 * Defining variables in dictionary and constructing path for data

 * Importing data

 * Manipulating background

 * Calculating PDF and constructing output-files

# FUTURE IMPROVEMENTS

 * Save interpolated data.

   * Implement HDF5 to save data.

 * Automatize data_magic.

 * Prevent background subtraction for returning negative values.

# BUGS

 * Sum doesn't work properbly

 * when all bg and data is the same, but bg and data differ in length

