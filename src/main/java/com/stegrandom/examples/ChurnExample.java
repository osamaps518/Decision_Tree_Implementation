package com.stegrandom.examples;

import com.stegrandom.model.DecisionTree;
import com.stegrandom.model.InformationTheoryMetrics;
import com.stegrandom.utils.DataLoader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;

public class ChurnExample {
  // File paths for our training and test data
  private static final String TRAIN_PATH = "src/main/resources/churn/train.csv";
  private static final String TEST_PATH = "src/main/resources/churn/test.csv";

  public static void main(String[] args) {
    try {
      // Load our training and test datasets
      System.out.println("Loading datasets...");
      DataLoader trainLoader = new DataLoader(TRAIN_PATH);
      DataLoader testLoader = new DataLoader(TEST_PATH);

      String[][] trainFeatures = trainLoader.load();
      String[][] testFeatures = testLoader.load();

      if (trainFeatures == null || testFeatures == null) {
        System.err.println("Error: Failed to load datasets");
        return;
      }

      // Print first few rows for verification
      System.out.println("First 3 rows of training data:");
      for (int i = 0; i < Math.min(3, trainFeatures.length); i++) {
        System.out.println(Arrays.toString(trainFeatures[i]));
      }

      // Separate features and target variables
      // Assuming the Churn (target) is the last column
      String[] trainTarget = extractTarget(trainFeatures, trainFeatures[0].length - 1);
      String[] testTarget = extractTarget(testFeatures, testFeatures[0].length - 1);

      // Print unique values in target variable
      System.out.println("Training set unique values in target: " +
          new HashSet<>(Arrays.asList(trainTarget)));
      System.out.println("Test set unique values in target: " +
          new HashSet<>(Arrays.asList(testTarget)));

      // Remove the target column from features
      trainFeatures = removeColumn(trainFeatures, trainFeatures[0].length - 1);
      testFeatures = removeColumn(testFeatures, testFeatures[0].length - 1);

      // Calculate and display information gain for each feature
      System.out.println("\nCalculating Information Gain for each feature:");
      for (int i = 0; i < trainFeatures[0].length; i++) {
        double gain = InformationTheoryMetrics.calculateInformationGain(i, trainFeatures, trainTarget);
        System.out.printf("Feature %d Information Gain: %.4f%n", i, gain);
      }

      // Initialize and train our decision tree model
      System.out.println("\nTraining decision tree model...");
      DecisionTree tree = new DecisionTree();

      String[] featureNames = { "Churn", "Status", "Plan", "Complains" };
      tree.setFeatureNames(featureNames);

      tree.fit(trainFeatures, trainTarget, 0); // Start at depth 0
      tree.printTree(); // Print the tree structure

      // Debug information about datasets
      System.out.println("\nDebug Information:");
      System.out.println("Test Features dimensions: " + testFeatures.length + " x " +
          (testFeatures.length > 0 ? testFeatures[0].length : 0));
      System.out.println("Test Target length: " + testTarget.length);

      // Make predictions
      System.out.println("\nMaking predictions...");
      String[] predictions = tree.predict(testFeatures);

      // Print first few predictions
      System.out.println("First few predictions:");
      for (int i = 0; i < Math.min(5, predictions.length); i++) {
        System.out.println("Prediction " + i + ": " + predictions[i]);
      }

      // Validate predictions before conversion
      boolean hasNulls = Arrays.stream(predictions).anyMatch(p -> p == null);
      if (hasNulls) {
        System.err.println("Warning: Null values detected in predictions!");
        for (int i = 0; i < predictions.length; i++) {
          if (predictions[i] == null) {
            System.err.println("Null prediction at index: " + i);
          }
        }
      }

      // Convert predictions and actual values to double arrays for metrics
      double[] actualValues = convertToDouble(testTarget);
      double[] predictedValues = convertToDouble(predictions);

      // Calculate and display metrics
      printMetrics(actualValues, predictedValues);

    } catch (IOException e) {
      System.err.println("Error reading data files: " + e.getMessage());
      e.printStackTrace();
    }
  }

  /**
   * Extracts the target variable column from the feature matrix
   */
  private static String[] extractTarget(String[][] features, int targetColumnIndex) {
    return Arrays.stream(features)
        .map(row -> row[targetColumnIndex])
        .toArray(String[]::new);
  }

  /**
   * Removes a column from the feature matrix
   */
  private static String[][] removeColumn(String[][] matrix, int colToRemove) {
    return Arrays.stream(matrix)
        .map(row -> {
          String[] newRow = new String[row.length - 1];
          System.arraycopy(row, 0, newRow, 0, colToRemove);
          System.arraycopy(row, colToRemove + 1, newRow, colToRemove, row.length - colToRemove - 1);
          return newRow;
        })
        .toArray(String[][]::new);
  }

  /**
   * Converts String labels to double values for metric calculations
   * Assumes binary classification with "1" as positive class
   */
  private static double[] convertToDouble(String[] labels) {
    return Arrays.stream(labels)
        .mapToDouble(label -> label.equals("1") ? 1.0 : 0.0)
        .toArray();
  }

  /**
   * Calculates and prints all relevant performance metrics
   */
  private static void printMetrics(double[] actual, double[] predicted) {
    System.out.println("\nModel Performance Metrics:");
    System.out.println("---------------------------");
    System.out.printf("Accuracy:  %.2f%%%n",
        InformationTheoryMetrics.calculateAccuracy(actual, predicted) * 100);
    System.out.printf("Precision: %.2f%%%n",
        InformationTheoryMetrics.calculatePrecision(actual, predicted) * 100);
    System.out.printf("Recall:    %.2f%%%n",
        InformationTheoryMetrics.calculateRecall(actual, predicted) * 100);
    System.out.printf("F-Score:   %.2f%%%n",
        InformationTheoryMetrics.calculateFScore(actual, predicted) * 100);
  }
}
