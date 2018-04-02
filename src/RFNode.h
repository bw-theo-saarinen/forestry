#ifndef FORESTRYCPP_RFNODE_H
#define FORESTRYCPP_RFNODE_H

#include <RcppEigen.h>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "DataFrame.h"

class RFNode {

public:
  RFNode();
  virtual ~RFNode();

  void setLeafNode(
    std::unique_ptr< std::vector<size_t> > averagingSampleIndex,
    std::unique_ptr< std::vector<size_t> > splittingSampleIndex
  );

  void setSplitNode(
    size_t splitFeature,
    double splitValue,
    std::unique_ptr< RFNode > leftChild,
    std::unique_ptr< RFNode > rightChild
  );

  void predict(
    std::vector<float> &outputPrediction,
    std::vector<size_t>* updateIndex,
    std::vector< std::vector<float> >* xNew,
    DataFrame* trainingData,
    Eigen::MatrixXf* weightMatrix
  );

  bool is_leaf();

  void printSubtree(int indentSpace=0);

  size_t getSplitFeature() {
    if (is_leaf()) {
      throw "Cannot get split feature for a leaf.";
    } else {
      return _splitFeature;
    }
  }

  double getSplitValue() {
    if (is_leaf()) {
      throw "Cannot get split feature for a leaf.";
    } else {
      return _splitValue;
    }
  }

  RFNode* getLeftChild() {
    if (is_leaf()) {
      throw "Cannot get left child for a leaf.";
    } else {
      return _leftChild.get();
    }
  }

  RFNode* getRightChild() {
    if (is_leaf()) {
      throw "Cannot get right child for a leaf.";
    } else {
      return _rightChild.get();
    }
  }

  size_t getSplitCount() {
    return _splitCount;
  }

  size_t getAverageCount() {
    return _averageCount;
  }

  std::vector<size_t>* getAveragingIndex() {
    return _averagingSampleIndex.get();
  }

private:
  std::unique_ptr< std::vector<size_t> > _averagingSampleIndex;
  std::unique_ptr< std::vector<size_t> > _splittingSampleIndex;
  size_t _splitFeature;
  double _splitValue;
  std::unique_ptr< RFNode > _leftChild;
  std::unique_ptr< RFNode > _rightChild;
  size_t _averageCount;
  size_t _splitCount;
};


#endif //FORESTRYCPP_RFNODE_H
