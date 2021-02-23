# Idea: high-level interface; a class/function where the inference scheme,
# kernel etc are defined by arguments. Similar to abc r package
# also add possibility for plots about the method, e.g. as a function of epsilon,
# number of data points in observed data if a function is used to generate - take
# an optional list with different N


# https://github.com/cran/abc/blob/master/R/abc.R

class ABC:

    def __init__(self, method="rejection", kernel="epkov"):
        pass

    # if(!any(method == c("rejection", "loclinear", "neuralnet","ridge"))){
    #    stop("Method must be rejection, loclinear, or neuralnet or ridge")
