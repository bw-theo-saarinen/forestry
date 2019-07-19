#include <Rcpp.h>
#include <vector>
#include <string>
#include <iostream>

void print_vector(
    std::vector<int> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
  }
  Rcpp::Rcout << std::endl;
}

void print_vector(
  std::vector<size_t> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
  }
  Rcpp::Rcout << std::endl;
}

void print_vector(
    std::vector<float> v
){
  for (auto i = v.begin(); i != v.end(); ++i){
    Rcpp::Rcout << *i << ' ';
  }
  Rcpp::Rcout << std::endl;
}
