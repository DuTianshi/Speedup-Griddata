# Speedup-Griddata
The scipy.interpolate provides the griddata function, however, it perform the grids relationship each time making it quite slow to employed it onto large data processes
So a seprate of the griddata function was made into to parts similar to the scipy.interpolate.interp1d functions, 
The first step is to establish the relationships between old and new grids,
the second step is to employ this established relationship onto different input data.
