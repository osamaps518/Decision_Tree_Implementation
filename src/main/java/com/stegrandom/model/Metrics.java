package com.stegrandom.model;

import com.stegrandom.core.Dataset;
import com.stegrandom.core.Feature;

public interface Metrics {
  // Information Theory Metrics
  double calculateEntropy(Dataset dataset);

  double calculateInformationGain(Dataset dataset, Feature feature);

  double calculateGainRatio(Dataset dataset, Feature feature);

  // Evaluation Metrics
  double calculateAccuracy(double[] actualValues, double[] predictedValues);

  double calculatePrecision(double[] actualValues, double[] predictedValues);

  double calculateRecall(double[] actualValues, double[] predictedValues);

  double calculateFScore(double[] actualValues, double[] predictedValues);
}