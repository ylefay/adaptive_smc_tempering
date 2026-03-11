# Pine saplings in Finland

See [R-spatstat documentation](https://search.r-project.org/CRAN/refmans/spatstat.data/html/finpines.html) for the
finpines dataset.
The csv file `finpines.csv` has been generated from the R dataset using the following code:

```r
# pines data extraction
library(spatstat)
data(finpines)
data_x <- (finpines$x + 5)/10 # normalize data to unit square
data_y <- (finpines$y + 8)/10
plot(x = data_x, y = data_y, type = "p", xlab = "x coordinate", ylab = "y coordinate")

df_pines = data.frame(data_x, data_y)
setwd("~/Dropbox/smc_hmc/python_smchmc/smc_sampler_functions")
write.csv(df_pines, file = "df_pines.csv")
```

See [Alexander Buchholz' repository](https://github.com/alexanderbuchholz/hsmc/blob/master/smc_sampler_functions/pines_data_r.r).