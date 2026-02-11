install.packages('icdpicr', lib = '.')
# Set the library path to the current directory
.libPaths("./")

# Load the package
library(icdpicr)

# Set the path to the dataset
dataset_path <- "data/interim/diagnoses_long.csv"

# Read the dataset
patients <- read.csv(dataset_path)

df <- cat_trauma(df = patients, dx_pre='ICD10_', icd10='base', i10_iss_method='roc_max_NIS', calc_method = 1, verbose = FALSE)

write.csv(df, 'data/interim/computed_iss_df.csv', row.names = FALSE)
