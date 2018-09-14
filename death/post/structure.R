# it feed the code structure into the net.

# you should run this after post.R and multiindex.R
# this script augments the dataset codes by inserting more codes
# by lowering the precision of the precise codes.

source("loadall.R")
require(stringr)

# this is a function that spawns multiple columns with one column
# depth refers to the maximal depth of a column code. E.g., A1B2C3 has depth 6.
# When code is A1B, we pad the code to have depth 6 then start splitting.


spawn <- function(dt, col, depth){
    dt <- get(dt)
    dt <- dt %>% mutate(!!col:=str_pad(dt[[col]],depth,side="right",pad="0"))
    dt
}