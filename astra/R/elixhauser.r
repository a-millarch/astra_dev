# Installs into a gitignored local library (.r_libs) so the package is not
# committed to the repository.
dir.create('.r_libs', showWarnings = FALSE)
.libPaths('.r_libs')
if (!requireNamespace('comorbidity', quietly = TRUE)) {
  install.packages('comorbidity', lib = '.r_libs', repos = 'https://cloud.r-project.org')
}

# Load the package
library(comorbidity)


# Set the path to the dataset
dataset_path <- "data/interim/pre_elix_df.csv"

# Read the dataset
patients <- read.csv(dataset_path)

elixhauser <- comorbidity(x = patients, id = "PID", code = "Diagnosekode", map = "elixhauser_icd10_quan", assign0 = TRUE
                          )

unw_eci <- score(elixhauser, weights = NULL, assign0 = FALSE)
eci <- score(elixhauser, weights = "vw", assign0 = TRUE)
all.equal(unw_eci, eci)
attr(eci,'map')
elixhauser$elixscore <-eci
write.csv(elixhauser, 'data/interim/computed_elix_df.csv', row.names = FALSE)
