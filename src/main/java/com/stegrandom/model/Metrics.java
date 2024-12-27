package com.stegrandom.model;

import com.stegrandom.core.Dataset;
import com.stegrandom.core.Feature;

public interface Metrics {
  // Information Theory Metrics
  double calculateEntropy(String[][] x, String[] y);

  // double calculateInformationGain(Dataset dataset, Feature feature);

  // First, extract all unique values
  // Then, Calculate Entropy for -before split-
  // Loop over all the dataset
  // Calculate Entropy for -after split-
  // calcualte difference between before and after
  // return information gain
  double calculateInformationGain(String[] dataset, double beforeSplitProba);

  double calculateGainRatio(Dataset dataset, Feature feature);

  double calculateEntropyAfterSplit(String[][] x, String[] y, int featureIndex);

  // Evaluation Metrics
  double calculateAccuracy(double[] actualValues, double[] predictedValues);

  double calculatePrecision(double[] actualValues, double[] predictedValues);

  double calculateRecall(double[] actualValues, double[] predictedValues);

  double calculateFScore(double[] actualValues, double[] predictedValues);
}