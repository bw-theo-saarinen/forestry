#include "multilayerForestry.h"
#include <random>
#include <thread>
#include <mutex>
#include "DataFrame.h"
#include "forestry.h"
#include "utils.h"
#include <RcppArmadillo.h>

multilayerForestry::multilayerForestry():
  _multilayerForests(nullptr), _gammas(0) {}

multilayerForestry::~multilayerForestry() {};

multilayerForestry::multilayerForestry(
  DataFrame* trainingData,
  size_t ntree,
  size_t nrounds,
  float eta,
  bool replace,
  size_t sampSize,
  float splitRatio,
  size_t mtry,
  size_t minNodeSizeSpt,
  size_t minNodeSizeAvg,
  size_t minNodeSizeToSplitSpt,
  size_t minNodeSizeToSplitAvg,
  float minSplitGain,
  size_t maxDepth,
  unsigned int seed,
  size_t nthread,
  bool verbose,
  bool splitMiddle,
  size_t maxObs,
  bool linear,
  bool gtBoost,
  float overfitPenalty,
  bool doubleTree
){
  this->_trainingData = trainingData;
  this->_ntree = ntree;
  this->_nrounds= nrounds;
  this->_eta = eta;
  this->_replace = replace;
  this->_sampSize = sampSize;
  this->_splitRatio = splitRatio;
  this->_mtry = mtry;
  this->_minNodeSizeAvg = minNodeSizeAvg;
  this->_minNodeSizeSpt = minNodeSizeSpt;
  this->_minNodeSizeToSplitAvg = minNodeSizeToSplitAvg;
  this->_minNodeSizeToSplitSpt = minNodeSizeToSplitSpt;
  this->_minSplitGain = minSplitGain;
  this->_maxDepth = maxDepth;
  this->_seed = seed;
  this->_nthread = nthread;
  this->_verbose = verbose;
  this->_splitMiddle = splitMiddle;
  this->_maxObs = maxObs;
  this->_linear = linear;
  this->_gtBoost = gtBoost;
  this->_overfitPenalty = overfitPenalty;
  this->_doubleTree = doubleTree;

  addForests(ntree);
}

// Helper to slice vector of indices [start:end] (inclusive)
std::vector<size_t> slice(std::vector<size_t> &v, int start, int end)
{
  std::vector<size_t> vec;
  std::copy(v.begin() + start, v.begin() + end + 1, std::back_inserter(vec));
  return vec;
}

float multilayerForestry::get_alpha(
    std::vector<size_t> sampleIndex,
    std::vector<float> accumulatedPredictions
){
  //return value of alpha (gradient tree boosting transfer) for particular leaf node

  // get original predicted mean
  size_t totalSampleSize = (sampleIndex).size();
  float accummulatedSum = 0;
  for (
      std::vector<size_t>::iterator it = (sampleIndex).begin();
      it != (sampleIndex).end();
      ++it
  ) {
    accummulatedSum += getTrainingData()->getOutcomePoint(*it);
  }
  float predicted_mean = accummulatedSum / totalSampleSize;

  // get residuals
  std::vector<float> residuals;
  for (
      std::vector<size_t>::iterator it = (sampleIndex).begin();
      it != (sampleIndex).end();
      ++it
  ) {
    residuals.push_back(getTrainingData()->getOutcomePoint(*it) - predicted_mean);
  }

  // get mean and stdev of residuals
  float sum = std::accumulate(residuals.begin(), residuals.end(), 0.0);
  float mean_r = sum / residuals.size();

  // Compute std deviation
  std::vector<float> diff(residuals.size());
  std::transform(residuals.begin(), residuals.end(), diff.begin(),
                 std::bind2nd(std::minus<double>(), mean_r));
  float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  float stdev_r = std::sqrt(sq_sum / diff.size());

  // populate keep_idx

  std::cout << "Mean res " << mean_r << " std res " << stdev_r << std::flush << std::endl;
  std::vector<size_t> keep_idx;
  for (
      std::vector<size_t>::iterator it = (sampleIndex).begin();
      it != (sampleIndex).end();
      ++it
  ) {
    std::vector<float> rowData(getTrainingData()->getNumColumns());
    getTrainingData()->getObservationData(rowData, *it);
    // if transfer point or correctly predicted base point (within 1 stdev of 0) keep point
    // assuming last feature in row indicates whether in transfer set
    if ((rowData[getTrainingData()->getNumColumns()-1]==1) || ((getTrainingData()->getOutcomePoint(*it) - predicted_mean)/stdev_r < 1)){
      keep_idx.push_back(*it);
    }
  }

  // analytically compute optimal value of alpha
  accummulatedSum = 0;
  for (
      std::vector<size_t>::iterator it = (sampleIndex).begin();
      it != (sampleIndex).end();
      ++it
  ) {

    // Get current row data
    std::vector<float> row_data(getTrainingData()->getNumColumns());
    getTrainingData()->getObservationData(row_data,
                                          *it);

    //CAST THE POInter nicesly
    std::vector< std::vector<float> >* current_observation = new std::vector< std::vector<float> >;


    for (const auto element : row_data) {
      std::vector<float> entry;
      entry.push_back(element);
      current_observation->push_back(entry);
    }

    // std::cout << "vector to predict is " << std::flush << std::endl;
    // print_vector(row_data);

    // Make outcome vector with orientation correct

    // std::cout << "Working fine up to " << (*it) << std::flush << std::endl;
    float y = getTrainingData()->getOutcomePoint(*it);
    // NEEDS TO BE CHANGED TO USE MULTIPLE PREDICTIONS
    accummulatedSum += y - accumulatedPredictions.at(*it);

    // std::cout << "Accumulated sum is " << accummulatedSum << std::flush << std::endl;
  }
  float alpha = accummulatedSum/keep_idx.size();

  return alpha;
}

// Returns naive sum of forest predictions up to iteration given, if iteration
// With base learner as avg of dataset, for all in sample data
std::vector<float> multilayerForestry::accumulated_predict(std::vector< forestry* > forests
) {

  std::unique_ptr< std::vector<float> > initialPrediction = forests[0]->predict(getTrainingData()->getAllFeatureData(),
                                                                                NULL,
                                                                                NULL,
                                                                                NULL);

  std::vector<float> prediction(initialPrediction->size(), getMeanOutcome());

  // Now loop through residual forests
  for (int i = 0; i < forests.size(); i ++) {

    std::unique_ptr< std::vector<float> > predictedResiduals =
      forests[i]->predict(getTrainingData()->getAllFeatureData(),
                          NULL,
                          NULL,
                          NULL);

    std::transform(prediction.begin(), prediction.end(),
                   predictedResiduals->begin(), prediction.begin(), std::plus<float>());
  }

  return prediction;
}


void multilayerForestry::addForests(size_t ntree) {

  // Create vectors to store gradient boosted forests and gamma values
  std::vector< forestry* > multilayerForests;
  std::vector<float> gammas(_nrounds);

  // Calculate initial prediction
  DataFrame *trainingData = getTrainingData();
  std::vector<float> outcomeData = *(trainingData->getOutcomeData());
  float meanOutcome =
    accumulate(outcomeData.begin(), outcomeData.end(), 0.0) / outcomeData.size();
  this->_meanOutcome = meanOutcome;

  std::vector<float> predictedOutcome(trainingData->getNumRows(), meanOutcome);

  // Apply gradient boosting to predict residuals
  std::vector<float> residuals(trainingData->getNumRows());
  for (int i = 0; i < getNrounds(); i++) {
    std::transform(outcomeData.begin(), outcomeData.end(),
                   predictedOutcome.begin(), residuals.begin(), std::minus<float>());

    std::shared_ptr< std::vector< std::vector<float> > > residualFeatureData_(
        new std::vector< std::vector<float> >(*(trainingData->getAllFeatureData()))
    );
    std::unique_ptr< std::vector<float> > residuals_(
        new std::vector<float>(residuals)
    );
    std::unique_ptr< std::vector<size_t> > residualCatCols_(
        new std::vector<size_t>(*(trainingData->getCatCols()))
    );
    std::unique_ptr< std::vector<size_t> > residualSplitCols_(
        new std::vector<size_t>(*(trainingData->getLinCols()))
    );
    std::unique_ptr< std::vector<size_t> > residualLinCols_(
        new std::vector<size_t>(*(trainingData->getLinCols()))
    );
    std::unique_ptr< std::vector<float> > residualSampleWeights_(
        new std::vector<float>(*(trainingData->getSampleWeights()))
    );
    std::unique_ptr< std::vector<float> > residualBootstrapWeights_(
        new std::vector<float>(*(trainingData->getBootstrapWeights()))
    );

    DataFrame* residualDataFrame = new DataFrame(
      residualFeatureData_,
      std::move(residuals_),
      std::move(residualCatCols_),
      std::move(residualSplitCols_),
      std::move(residualLinCols_),
      trainingData->getNumRows(),
      trainingData->getNumColumns(),
      std::move(residualSampleWeights_),
      std::move(residualBootstrapWeights_)
    );

    forestry *residualForest = new forestry(
      residualDataFrame,
      _ntree,
      _replace,
      _sampSize,
      _splitRatio,
      _mtry,
      _minNodeSizeSpt,
      _minNodeSizeAvg,
      _minNodeSizeToSplitSpt,
      _minNodeSizeToSplitAvg,
      _minSplitGain,
      _maxDepth,
      _seed,
      _nthread,
      _verbose,
      _splitMiddle,
      _maxObs,
      _linear,
      _overfitPenalty,
      _doubleTree
    );


    std::cout << "Just trained residual forest # " << i << std::flush << std::endl;

    multilayerForests.push_back(residualForest);
    /* Here add the step which replaces leaf node values with correct alphas */

    /*  Steps
     *
     *  1) Get vectors of leaf observations
     *
     *  2) Pass to function to calculate alphas and assign to nodes
     *
     *  3) Modify predict to return alpha correctly
     *
     */


    // ADD FLAG FOR GRADIENT TREE BOOSTING

    if (getgtBoost()) {
      std::cout << "Gradient tree boosting is on, so calculating alphas for " << i << std::flush << std::endl;
      // Get leaf node ID's from residual forest
      std::unique_ptr< std::vector<tree_info> > residual_forest_dta(
          new std::vector<tree_info>
      );

      residualForest->fillinTreeInfo(residual_forest_dta);

      std::vector< std::vector<size_t> > all_split_ids;

      for (size_t k = 0; k < 1/*residual_forest_dta->size()*/; k++) {

        // Get node sizes vector from tree data
        std::vector<int> split_ids = ((*residual_forest_dta)[k]).var_id;
        // Note we are using the averaging indices here
        std::vector<int> avg_ids_int = ((*residual_forest_dta)[k]).leafAveidx;

        // As we had indices in R, they were not zero indexed, so transform
        for (auto& id : avg_ids_int) {
          id--;
        }

        std::vector<size_t> avg_ids(avg_ids_int.size());


        for (size_t j = 0; j < avg_ids_int.size(); j++) {
          avg_ids[j] = (size_t) avg_ids_int[j];
        }

        std::vector<int> node_sizes;
        std::vector<int> node_sizes_filtered;
        for(const auto & id : split_ids) {
          if(id < 0) {
            node_sizes.push_back(-id);
          }
        }
        std::cout << "Avg ids " << std::endl;
        print_vector(avg_ids);
        std::cout << "split ids " << std::endl;
        print_vector(split_ids);
        std::cout << "Node sizes " << std::endl;
        print_vector(node_sizes);


        // Filter every other one if neg as we have duplicates
        for (size_t j = 1; j < node_sizes.size(); j += 2) {
          node_sizes_filtered.push_back(node_sizes[j]);
        }

        std::cout << "Filtered Node sizes " << std::endl;
        print_vector(node_sizes_filtered);


        // Vector of vectors of node averaging indices to pass to getAlpha function
        std::vector< std::vector<size_t> > node_indices(node_sizes_filtered.size());

        size_t start_idx = 0;
        for (size_t curr_node = 0; curr_node < node_sizes_filtered.size(); curr_node++) {
            std::vector<size_t> node_i_ids = slice(avg_ids,
                                                   start_idx,
                                                   start_idx + node_sizes_filtered[curr_node] - 1);

            std::cout << "Pushing back this vector to nodesizes" << std::endl;
            print_vector(node_i_ids);


            node_indices.at(curr_node) = node_i_ids;
            start_idx += node_sizes_filtered[curr_node];
        }


        // current iteration is i
        // std::vector< forestry* > multilayerForests(_nrounds);
        // Now calculate getAlpha of each subset of averaging indices
        std::vector<float> previousPredictions = accumulated_predict(multilayerForests);

        std::vector<float> node_alphas(node_indices.size());

        for (size_t j = 0; j < node_alphas.size(); j++) {
          node_alphas[j] = get_alpha(node_indices[j],
                                     previousPredictions);

          std::cout << "Made alpha " << j << " is " << node_alphas[j] << std::flush << std::endl;
        }

        std::cout << std::endl <<  "Node alphas :" << std::endl;
        print_vector(node_alphas);

        std::vector<float> modifiedOutcomeData(avg_ids.size());

        for (size_t j = 0; j < node_indices.size(); j++) {
          std::vector<size_t> currentIndices = node_indices[j];
          float currentAlpha = node_alphas[j];
          for (const auto& index : currentIndices) {
            modifiedOutcomeData[index] = currentAlpha;
            std::cout << "Just set modified outcome index " << index << " to " << currentAlpha << std::flush << std::endl;
          }
        }

        // Now set residual dataframe output column to modified outcome vector containing alphas

        std::cout << "going to set outcome data now" << std::flush << std::endl;

        multilayerForests[i]->getTrainingData()->setOutcomeData(modifiedOutcomeData);

      }

    } // END GT Boost step

    std::cout << "Done with GT Boosting step- predicting new residuals " << std::flush  << std::endl;

    std::unique_ptr< std::vector<float> > predictedResiduals = residualForest->predict(getTrainingData()->getAllFeatureData(),
                                                                                       NULL,
                                                                                       NULL,
                                                                                       NULL);




      // Calculate and store best gamma value
    // std::vector<float> bestPredictedResiduals(trainingData->getNumRows());
    // float minMeanSquaredError = INFINITY;
    // static inline float computeSquare (float x) { return x*x; }

    // for (float gamma = -1; gamma <= 1; gamma += 0.02) {
    //   std::vector<float> gammaPredictedResiduals(trainingData->getNumRows());
    //   std::vector<float> gammaError(trainingData->getNumRows());
    //
    //   // Find gamma with smallest mean squared error
    //   std::transform(predictedResiduals->begin(), predictedResiduals->end(),
    //                  gammaPredictedResiduals.begin(), std::bind1st(std::multiplies<float>(), gamma));
    //   std::transform(predictedOutcome->begin(), predictedOutcome->end(),
    //                  gammaPredictedResiduals.begin(), gammaError.begin(), std::plus<float>());
    //   std::transform(outcomeData->begin(), outcomeData->end(),
    //                  gammaError.begin(), gammaError.begin(), std::minus<float>());
    //   std::transform(gammaError.begin(), gammaError.end(), gammaError.begin(), computeSquare);
    //   float gammaMeanSquaredError =
    //     accumulate(gammaError.begin(), gammaError.end(), 0.0)/gammaError.size();
    //   std::cout << gammaMeanSquaredError << std::endl;
    //
    //   if (gammaMeanSquaredError < minMeanSquaredError) {
    //
    //     gammas[i] = (gamma * eta);
    //     minMeanSquaredError = gammaMeanSquaredError;
    //     bestPredictedResiduals = gammaPredictedResiduals;
    //   }
    // }

    gammas[i] = 1 * _eta;

    // Update prediction after each round of gradient boosting
    std::transform(predictedOutcome.begin(), predictedOutcome.end(),
                   predictedResiduals->begin(), predictedResiduals->end(), std::plus<float>());
  }

  // Save vector of forestry objects and gamma values
  std::unique_ptr<std::vector< forestry* > > multilayerForests_(
    new std::vector< forestry* >(multilayerForests)
  );

  this->_multilayerForests = std::move(multilayerForests_);
  this->_gammas = std::move(gammas);
}

std::unique_ptr< std::vector<float> > multilayerForestry::predict(
    std::vector< std::vector<float> >* xNew,
    arma::Mat<float>* weightMatrix
) {
  std::vector< forestry* > multilayerForests = *getMultilayerForests();
  std::vector<float> gammas = getGammas();

  std::unique_ptr< std::vector<float> > initialPrediction =
    multilayerForests[0]->predict(xNew,
                                  weightMatrix,
                                  NULL,
                                  NULL);

  std::vector<float> prediction(initialPrediction->size(), getMeanOutcome());

  // Use forestry objects and gamma values to make prediction
  for (int i = 0; i < getNrounds(); i ++) {
    std::unique_ptr< std::vector<float> > predictedResiduals =
      multilayerForests[i]->predict(xNew,
                                    weightMatrix,
                                    NULL,
                                    NULL);

    std::transform(predictedResiduals->begin(), predictedResiduals->end(),
                   predictedResiduals->begin(), std::bind1st(std::multiplies<float>(), gammas[i]));

    std::transform(prediction.begin(), prediction.end(),
                   predictedResiduals->begin(), prediction.begin(), std::plus<float>());
  }

  std::unique_ptr< std::vector<float> > prediction_ (
      new std::vector<float>(prediction)
  );

  return prediction_;
}

