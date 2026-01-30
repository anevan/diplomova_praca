import kagglehub

# Run this file to download datasets from different kaggle sources

# path = kagglehub.dataset_download("catiateixeira/wordwide-pm-polution-and-related-mortality")
# path = kagglehub.dataset_download("himanshunakrani/iris-dataset")
# path = kagglehub.dataset_download("hurshd0/abalone-uci")

# more info about this one: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
# path = kagglehub.dataset_download("sohommajumder21/appliances-energy-prediction-data-set")

path = kagglehub.dataset_download("dskagglemt/real-estate-valuation-by-uci")

# path = kagglehub.dataset_download("uciml/red-wine-quality-cortez-et-al-2009")
# path = kagglehub.dataset_download("uciml/autompg-dataset")
# path = kagglehub.dataset_download("andrewmvd/heart-failure-clinical-data")
# path = kagglehub.dataset_download("rishidamarla/heart-disease-prediction")
# path = kagglehub.dataset_download("yasserh/breast-cancer-dataset")

print("Path to dataset files:", path)
