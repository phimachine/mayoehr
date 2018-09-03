# you should run this after post.R and multiindex.R
# this script augments the dataset codes by inserting more codes
# by lowering the precision of the precise codes.

source("loadall.R")

uniquebarsep <- function(dt, col){
    hello<-unlist(strsplit(dt[[col]],split="\\|"))
    hello<-unique(unlist(hello))
}
