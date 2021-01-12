#include<RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

/*
 * Input:
 *    D: data, each column represent one point, the first
 *      row indicates the category(0-1), the other rows 
 *      indicate the attributes.
 *    IR: the proportion of the number of majority and minority, determining the
 *        # of synthetic samples
 *    k : the # of nearest neighbors
 */

// [[Rcpp::export]]
Rcpp::List div_data(arma::mat D){
  /*  divide the original data into two categories
   *   INPUT: 
   *       D:all data with labels in the 1st row
   *   OUTPUT:
   *       P: positive data with no labels
   *       N: negative data with no labels
   *       majortity type: for function KNN_P
   */
  mat P,N;
  int points = D.n_cols, attr = D.n_rows - 1;
  int type_p, n_p, n_t1 = sum(D.row(0));
  int p_ind = 0, n_ind = 0;
  
  // fine minority and # of minority points
  type_p = (n_t1 < points - n_t1)? 1:0;
  n_p = (n_t1 < points - n_t1)? n_t1:(points-n_t1); 
  // storage minority and majority points in P,N, respectively
  P.set_size(attr, n_p);
  N.set_size(attr, points-n_p);
  for(int i=0; i<points; i++){
    if(D(0,i) == type_p){
      P.col(p_ind) = D(span(1,attr),i);// ignore the first row(type data)
      p_ind++;}
    else{
      N.col(n_ind) = D(span(1,attr),i);
      n_ind++;
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("P") = P,
                            Rcpp::Named("N") = N,
                            Rcpp::Named("majority type") = 1-type_p);
}


// [[Rcpp::export]]
arma::vec KNN_P(arma::mat P,arma::mat D, int k, int type_maj){
  /* INPUT:
   *    P: Positive(minority) data
   *    D: in which find nearest points
   *    k: # of nearest neighbors, smaller than # of points in D!
   * OUTPUT:
   *    n_maj: # of majority nearest neighbors of each minority point
   */
  arma::vec n_maj(P.n_cols), all_dist(D.n_cols), type(D.n_cols);
  arma::vec dist_sort_ind;
  int u;
  
  if(k>D.n_cols-1) k = D.n_cols-1; // k can't be too large!
  type = D.row(0).t();
  D.shed_row(0);
  
  for(int i=0; i<P.n_cols; i++){ // ith minority point
    // calculate distances from points in D, sorted in ascending order
    all_dist = zeros(D.n_cols);
    all_dist = sum(square(D.each_col() - P.col(i))).t();
    dist_sort_ind = conv_to<vec>::from(sort_index(all_dist, "ascend"));// corresponding index
    
    // identify the type of first k distances
    u = 0;
    for(int j=1; j<=k; j++){ // the 0th: itself
      if(type(dist_sort_ind(j))==type_maj) u++;
    }
    n_maj(i) = u;
  }
  return n_maj;
}


// [[Rcpp::export]]
Rcpp::List remove_noise(arma::vec n_maj, int k, arma::mat P){
  // we only need the pureP
  arma::uvec noise_ind(n_maj.n_elem, fill::zeros);
  arma::mat pureP = P;
  int count = 0;
  // detect noise
  for(int i=0; i<n_maj.n_elem; i++){
    if(n_maj(i)==k){
      noise_ind(count) = i;
      count++;
    }
  }
  // remove noise information
  if(count>0){
    n_maj.shed_rows(noise_ind(span(0,count-1)));
    pureP.shed_cols(noise_ind(span(0,count-1)));
  }
  return Rcpp::List::create(Rcpp::Named("number of majority neighbors") = n_maj,
                            Rcpp::Named("pure P") = pureP);
}


//[[Rcpp::export]]
arma::vec absolute_density(arma::mat P, arma::mat Neigh, int k,  bool self = true){
  /* 
   * INPUT:
   *    P: points to calculate absolute density(according to knn)
   *    Neigh: points in which to find neighbors
   *    self: P==Neigh?
   *    k: # of nearest neighbors, smaller than # of points in Neigh
   *    
   *  OUTPUT:
   *    AD: absolute density of every point in P
   */
  arma::vec AD = zeros(P.n_cols);
  arma::vec all_dist;
  int indicator = 0;
  
  if(self ==false) indicator = 1;
  if(k>(Neigh.n_cols-1+indicator)) k = Neigh.n_cols - 1 + indicator;
  for(int i=0; i<P.n_cols; i++){ // ith point
    // calculate distances, sorted in ascending order
    all_dist = zeros(Neigh.n_cols);
    all_dist = sum(square(Neigh.each_col() - P.col(i))).t();
    all_dist = sort(sqrt(all_dist), "ascend");
    AD(i) = sum(all_dist(span(1-indicator,k-indicator)));
  }
  return AD;
}


// [[Rcpp::export]]
arma::vec relative_dentisy(arma::mat P, arma::mat N, int k){
  arma::vec RD(P.n_cols), AD_hon, AD_hen;
  
  AD_hon = absolute_density(P, P, k);
  AD_hen = absolute_density(P, N, k, false);
  RD = AD_hen / AD_hon;
  return RD;
}


// [[Rcpp::export]]
arma::mat KNN(arma::mat P,arma::mat D, int k){
  /* Params:
   *    k: # of nearest neighbors, smaller than # of points in D!
   * OUTPUT:
   *    nnarray: for each point in P, find its k neighbors' indexes in D (
   *    index starts with 0)
   */
  if(k>D.n_cols-1) k = D.n_cols-1; // k can't be too large!
  
  arma::mat nnarray(P.n_cols, k);
  for(int i=0; i<P.n_cols; i++){ // ith minority point
    // calculate distances, sorted in ascending order
    arma::vec all_dist = zeros(D.n_cols);
    all_dist = sum(square(D.each_col() - P.col(i))).t();
    // corresponding index
    arma::vec dist_sort_ind = conv_to<vec>::from(sort_index(all_dist, "ascend"));
    
    nnarray.row(i) = dist_sort_ind(span(1, k)).as_row(); // 0th: itself
  }
  return nnarray;
  
}


// [[Rcpp::export]]
arma::mat clusterRD(arma::mat RD){
  /*
   * Params:
   *    RD: 1st col is the sample indexes,  2nd col is the relative densities
   * return:
   *    cluster_result: 0 for border sample, 1 for safe sample
   */
  int n = RD.n_rows;
  arma::mat centers;
  // return centers of two clusters
  kmeans(centers, RD.col(1).as_row(), 
         2, random_subset, 10, false);
  
  double c1 = centers(0,0), c2 = centers(0,1);
  double threshold = (c1 + c2)/2;
  
  arma::mat cluster_result(size(RD));
  cluster_result.col(0) = RD.col(0);
  for (int i=0;i<n;i++){
    if (RD(i,1) < threshold){
      cluster_result(i,1) = 0; // border point
    }
    else cluster_result(i,1) = 1; //safe point
  }
  return cluster_result;
}


// [[Rcpp::export]]
Rcpp::List over_sampling(int N, int k,
                         arma::mat &pureP,
                         arma::mat RD_clusters,
                         arma::mat nnarray,
                         arma::mat num_maj){
  /*
   * Params:
   *    N: num of total synthetic samples to be generated
   *    k: k neighbors
   *    pureP: minority samples without noise
   *    RD_clusters: cluster_result, 0 for border sample, 1 for safe sample
   *    nnarray: k neighbors' indexes of each sample in pureP
   *    num_maj: num of majority samples of each sample in pureP
   * return:
   *    the synthetic samples
   */
  
  arma::uvec safe_pts = find(RD_clusters.col(1)==1); // safe points' indexes
  arma::uvec border_pts = find(RD_clusters.col(1)==0); // border points' indexes
  
  arma::mat synthetic(pureP.n_rows, N, fill::zeros); // generate N synthetic samples
  
  
  int n_safe_pts = safe_pts.n_elem;
  int n_border_pts = border_pts.n_elem;
  // number of total samples to be generated in safe samples and border samples
  int N_safe = round(N * n_safe_pts/(n_safe_pts + n_border_pts));
  int N_border = N - N_safe;
  
  // weights for samples in pureP
  arma::vec weights = (k - num_maj.as_col())/(k+1);
  weights(safe_pts) = weights(safe_pts) / sum(weights(safe_pts));
  weights(border_pts) = weights(border_pts) / sum(weights(border_pts));
  
  // number of samples to be generated around sample i
  arma::vec ni_safe = floor(weights(safe_pts) * N_safe);
  arma::vec ni_border = floor(weights(border_pts) * N_border);
  
  /*
   * note: due to flooring,
   * sum(ni_safe) + sum(ni_border) <= N
   * if sum(ni_safe) + sum(ni_border) < N,
   * we generate one more synthetic sample(s) around safe sample i
   * according to its weight
   */
  int numdif = N - sum(ni_safe) - sum(ni_border);
  if (numdif > 0){
    arma::uvec idx = Rcpp::RcppArmadillo::sample(regspace<uvec>(0, ni_safe.n_elem-1),
                                           numdif, true, weights(safe_pts));
    /*
     * sampling with replacement may get multiple same index
     * e.g. 
     * if idx = (1,1,2,5,1), 
     * then the result of code "ni_safe(idx) += 1" is:
     * ni_safe[i] += 3, ni_safe[2] += 1, ni_safe[5] += 1
     * 
     * so we can get exactly dif_safe more samples
     */
    ni_safe(idx) += 1;
  }
  
  
  int num_syn = 0; // number of already generated samples
  /*
   * oversampling in safe samples
   */ 
  for (int i=0; i<n_safe_pts; i++){
    unsigned int idx = safe_pts(i);
    arma::rowvec nnarray_i = nnarray.row(idx); // neighbors of sample i
    
    // number of samples to be generated around sample i
    int n_i = ni_safe(i);
    if (n_i>0){
      // arma_rng::set_seed_random(2020);
      arma::ivec nn = randi<ivec>(n_i, distr_param(0, k-1)); // randomly choose neighbors
      for (int m=0; m<n_i; m++){
        double alpha = randu();
        
        int neigh = nnarray_i(nn(m));
        synthetic.col(num_syn) = (1-alpha) * pureP.col(idx) + alpha * pureP.col(neigh);
        
        num_syn += 1;
      }
    }
  }
  
  /*
   * oversampling in border samples
   */ 
  for (int j=0; j<n_border_pts; j++){
    unsigned int idx = border_pts(j);
    
    arma::rowvec nnarray_i = nnarray.row(idx); // neighbors of sample j
    // number of samples to be generated around sample i
    int n_i = ni_border(j);
    if (n_i>0){
      // arma_rng::set_seed_random(2020);
      arma::ivec nn = randi<ivec>(n_i, distr_param(0, k-1)); // randomly choose neighbors
      for (int m=0; m<n_i; m++){
        double alpha = randu();
        
        int neigh = nnarray_i(nn(m));
        synthetic.col(num_syn) = (1-alpha) * pureP.col(idx) + alpha * pureP.col(neigh);
        
        num_syn += 1;
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("synthetic") = synthetic,
                            Rcpp::Named("num_syn") = num_syn);
}





