uniquebarsep <- function(dt, col){
    hello<-unlist(strsplit(dt[[col]],split="\\|"))
    hello<-unique(unlist(hello))
}
