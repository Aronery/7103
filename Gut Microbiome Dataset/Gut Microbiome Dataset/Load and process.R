library(readxl)

#import the related sample metadata
runs_associated_with_D006262 = read_excel("runs_associated_with_D006262.xlsx")
runsID_associated_with_D006262 = runs_associated_with_D006262$`run ID`
runs_associated_with_D015179 = read_excel("runs_associated_with_D015179.xlsx")
runsID_associated_with_D015179 = runs_associated_with_D015179$`run ID`

#import the related species metadata
CRC_species = read.csv("CRC_species.csv")
species_list_CRC = unique(CRC_species$NCBI.taxon.id[which(CRC_species$median.relative.abundance>=0.01)])
species_list = unique(species_list_CRC)

#import the species abundance matrices
abundance_mat_species_D006262_CRC = read.csv("abundance_mat_species_D006262_CRC.csv",header = T)[,-1]
abundance_mat_species_D065626_CRC = read.csv("abundance_mat_species_D015179_CRC.csv",header = T)[,-1]
dim(abundance_mat_species_D006262_CRC)
dim(abundance_mat_species_D065626_CRC)

#Manually write the label for each sample, 0 for Health, 1 for CRC
Sample_label = c(rep(0,dim(abundance_mat_species_D006262_CRC)[1]),rep(1,dim(abundance_mat_species_D065626_CRC)[1]))
dat = cbind(mat,label)
loc = which(rowSums(mat) == 0)
dat = dat[-loc,]
colnames(dat) = c(species_list,"label")

#Using "dat" for modeling, where the last column are the labels