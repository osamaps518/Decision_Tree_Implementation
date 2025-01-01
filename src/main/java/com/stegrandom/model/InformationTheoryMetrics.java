package com.stegrandom.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class InformationTheoryMetrics {

  // Calculate entropy for a dataset
  public static double calculateEntropy(String[][] x, String[] y) {
    Map<String, Integer> labelCounts = new HashMap<>();
    for (String label : y) {
      labelCounts.put(label, labelCounts.getOrDefault(label, 0) + 1);
    }

    double entropy = 0.0;
    int total = y.length;
    for (int count : labelCounts.values()) {
      double probability = (double) count / total;
      entropy -= probability * (Math.log(probability) / Math.log(2));
    }
    return entropy;
  }

  // Calculate information gain
  public static double calculateInformationGain(int featureIndex, String[][] features, String[] target) {
    double beforeSplitEntropy = calculateEntropy(features, target);

    Map<String, List<String>> subsets = new HashMap<>();
    for (int i = 0; i < features.length; i++) {
      String featureValue = features[i][featureIndex];
      subsets.putIfAbsent(featureValue, new ArrayList<>());
      subsets.get(featureValue).add(target[i]);
    }

    double afterSplitEntropy = 0.0;
    int totalSamples = target.length;
    for (List<String> subset : subsets.values()) {
      String[] subsetLabels = subset.toArray(new String[0]);
      double subsetProbability = (double) subset.size() / totalSamples;
      afterSplitEntropy += subsetProbability * calculateEntropy(new String[0][0], subsetLabels);
    }

    return beforeSplitEntropy - afterSplitEntropy;
  }

  // Calculate entropy after a split
  public static double calculateEntropyAfterSplit(String[][] x, String[] y, int featureIndex) {
    Map<String, List<String>> subsets = new HashMap<>();

    for (int i = 0; i < x.length; i++) {
      String featureValue = x[i][featureIndex];
      subsets.putIfAbsent(featureValue, new ArrayList<>());
      subsets.get(featureValue).add(y[i]);
    }

    double totalEntropy = 0.0;
    int totalSamples = y.length;
    for (List<String> subset : subsets.values()) {
      String[] subsetLabels = subset.toArray(new String[0]);
      double subsetProbability = (double) subset.size() / totalSamples;
      totalEntropy += subsetProbability * calculateEntropy(new String[0][0], subsetLabels);
    }
    return totalEntropy;
  }

  // Calculate accuracy
  public static double calculateAccuracy(double[] actualValues, double[] predictedValues) {
    int correct = 0;
    for (int i = 0; i < actualValues.length; i++) {
      if (actualValues[i] == predictedValues[i])
        correct++;
    }
    return (double) correct / actualValues.length;
  }

  // Calculate precision
  public static double calculatePrecision(double[] actualValues, double[] predictedValues) {
    int truePositive = 0, falsePositive = 0;
    for (int i = 0; i < actualValues.length; i++) {
      if (predictedValues[i] == 1) {
        if (actualValues[i] == 1)
          truePositive++;
        else
          falsePositive++;
      }
    }
    return (double) truePositive / (truePositive + falsePositive);
  }

  // Calculate recall
  public static double calculateRecall(double[] actualValues, double[] predictedValues) {
    int truePositive = 0, falseNegative = 0;
    for (int i = 0; i < actualValues.length; i++) {
      if (actualValues[i] == 1) {
        if (predictedValues[i] == 1)
          truePositive++;
        else
          falseNegative++;
      }
    }
    return (double) truePositive / (truePositive + falseNegative);
  }

  // Calculate F-Score
  public static double calculateFScore(double[] actualValues, double[] predictedValues) {
    double precision = calculatePrecision(actualValues, predictedValues);
    double recall = calculateRecall(actualValues, predictedValues);
    return 2 * (precision * recall) / (precision + recall);
  }

}