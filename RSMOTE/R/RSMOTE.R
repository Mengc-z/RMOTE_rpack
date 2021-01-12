#' @title RSMOTE, an over-sampler for imbalanced data
#' 
#' @description Generate synthetic samples for imbalanced data with two classes,using RSMOTE algorithm. Synthetic samples are generated from minority class.
#' 
#' @details Data: each column represents one sample, the first element represent its class;
#' the number of synthetic samples = IR * number of majority samples - number of minority samples.
#' 
#' @param Data A data frame or matrix of numeric-attributed dataset,with the first row indicating the two classes with 0 and 1.
#' @param IR The proportion of minority and majority samples.
#' @param k The number of nearest neighbors during sampling process.
#' 
#' @return The synthetic samples as a matrix.
#' @export
#' @examples 
#' library(RSMOTE)
#' mat1 = matrix(nrow=3, ncol=1000)
#' mat1[1,] = 1
#' mat1[2,] = c(rnorm(1000,1,0.5))
#' mat1[3,] = c(rnorm(1000,0.6,0.6))
#' mat2 = matrix(nrow=3, ncol=100)
#' mat2[1,] = 0
#' mat2[2,] = c(rnorm(100,2.5,0.7))
#' mat2[3,] = c(rnorm(100,2.5,0.5))
#' mat = cbind(mat1,mat2)
#' synthetic = RSMOTE(mat, IR = 0.9, k = 5)
#' library(ggplot2)
#' ggplot()+
#' geom_point(aes(mat2[2,],mat2[3,]),col='blue')+
#' geom_point(aes(mat1[2,],mat1[3,]),col='red')+
#' geom_point(aes(synthetic[1,], synthetic[2,]), col='green')
RSMOTE <- function(Data, IR = 0.8, k = 6){
  # check input parameters
  if(length(unique(Data[1,])) != 2) 
    return("only generate synthetic samples for imbalanced data with two catogeries")
  if(!(all(unique(Data[1,])==c(1,0)) | all(unique(Data[1,])==c(0,1))))
    return("the first row of data must be 0 or 1, representing two catogeries")
  if(IR<0) IR = -IR
  else if(IR>1) IR = 1
  if(k>=ncol(Data)) k = ncol(Data) / 2
  
  # divide data into majority(N) and minority(P)
  divData = div_data(Data)
  P = divData$P # minority
  N = divData$N
  type = divData$'majority type'
  
  # calculate the number of majority nearest neighbors
  num_maj = KNN_P(P, Data, k, type)#$"number of majority neighbors"
  
  # remove noise
  removeNoise = remove_noise(num_maj, k, P)
  pureP = removeNoise$'pure P'
  num_maj = removeNoise$"number of majority neighbors"
  
  # calculate the number of synthetic samples in need
  num_syn = ncol(N) * IR - ncol(pureP)
  
  # calculate relative density
  # 1st col: the indexes of samples in pureP; 2nd col: the corresponding RD values
  RD = matrix(nrow=ncol(pureP),ncol=2)
  RD[,1] = c(0:(ncol(pureP)-1))
  RD[,2] = relative_dentisy(pureP, N, k)
  RD_clusters = clusterRD(RD)
  
  # for each minority point, 
  # find the indexes of k nearest neighbors in minority sets
  nnarray = KNN(pureP, pureP, k)
  
  # oversampling
  result = over_sampling(num_syn, k, pureP, RD_clusters, nnarray, num_maj)
  synthetic = result$synthetic
  
  return(synthetic)
}

