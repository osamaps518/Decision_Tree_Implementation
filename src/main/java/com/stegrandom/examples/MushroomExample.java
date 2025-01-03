package com.stegrandom.examples;

import com.stegrandom.model.DecisionTree;
import com.stegrandom.model.InformationTheoryMetrics;
import com.stegrandom.utils.DataLoader;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class MushroomExample {
  // File paths for our training and test data
  private static final String TRAIN_PATH = "src/main/resources/mushroom/train.csv";
  private static final String TEST_PATH = "src/main/resources/mushroom/test.csv";

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
      System.out.println("First 3 rows of training data:");
      for (int i = 0; i < 3; i++) {
        System.out.println(Arrays.toString(trainFeatures[i]));
      }
      Set<String> allFeature4Values = new HashSet<>();
      for (String[] row : trainFeatures) {
        allFeature4Values.add(row[4]);
      }
      System.out.println("All unique values for feature 4: " + allFeature4Values);

      // Separate features and target variables
      // Assuming the target (edible/poisonous) is the first column
      String[] trainTarget = extractTarget(trainFeatures, 0);
      String[] testTarget = extractTarget(testFeatures, 0);

      System.out.println("Training set unique values in target: " +
          new HashSet<>(Arrays.asList(trainTarget)));
      System.out.println("Test set unique values in target: " +
          new HashSet<>(Arrays.asList(testTarget)));

      Set<String> trainOdors = new HashSet<>();
      Set<String> testOdors = new HashSet<>();
      for (String[] row : trainFeatures) {
        trainOdors.add(row[4]); // Feature 4 is odor
      }
      for (String[] row : testFeatures) {
        testOdors.add(row[4]);
      }
      System.out.println("Training set unique odors: " + trainOdors);
      System.out.println("Test set unique odors: " + testOdors);

      // Remove the target column from features
      trainFeatures = removeColumn(trainFeatures, 0);
      testFeatures = removeColumn(testFeatures, 0);

      // Initialize and train our decision tree model
      System.out.println("Training decision tree model...");
      DecisionTree tree = new DecisionTree();
      tree.fit(trainFeatures, trainTarget, 0); // Start at depth 0
      tree.printTree(); // Print the tree structure

      // Add debugging information about our datasets
      System.out.println("\nDebug Information:");
      System.out.println("Test Features dimensions: " + testFeatures.length + " x " +
          (testFeatures.length > 0 ? testFeatures[0].length : 0));
      System.out.println("Test Target length: " + testTarget.length);

      // Make predictions on test set
      System.out.println("\nMaking predictions...");
      String[] predictions = tree.predict(testFeatures);

      // Debug predictions
      System.out.println("Number of predictions: " + predictions.length);
      System.out.println("First few predictions:");
      for (int i = 0; i < Math.min(5, predictions.length); i++) {
        System.out.println("Prediction " + i + ": " + predictions[i]);
      }

      // Validate predictions before conversion
      boolean hasNulls = Arrays.stream(predictions).anyMatch(p -> p == null);
      if (hasNulls) {
        System.err.println("Warning: Null values detected in predictions!");
        // Print positions of null values
        for (int i = 0; i < predictions.length; i++) {
          if (predictions[i] == null) {
            System.err.println("Null prediction at index: " + i);
          }
        }
      }

      // Convert predictions and actual values to double arrays for metrics
      double[] actualValues = convertToDouble(testTarget);
      double[] predictedValues = convertToDouble(predictions);

      // Calculate and display performance metrics
      printMetrics(actualValues, predictedValues);

    } catch (IOException e) {
      System.err.println("Error reading data files: " + e.getMessage());
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
   * Assumes binary classification with "edible" as positive class
   */
  private static double[] convertToDouble(String[] labels) {
    return Arrays.stream(labels)
        .mapToDouble(label -> label.equals("e") ? 1.0 : 0.0)
        .toArray();
  }

  /**
   * Calculates and prints all relevant performance metrics
   */
  private static void printMetrics(double[] actual, double[] predicted) {
    System.out.println("\nModel Performance Metrics:");
    System.out.println("---------------------------");
    System.out.printf("Accuracy:  %.2f%%\n",
        InformationTheoryMetrics.calculateAccuracy(actual, predicted) * 100);
    System.out.printf("Precision: %.2f%%\n",
        InformationTheoryMetrics.calculatePrecision(actual, predicted) * 100);
    System.out.printf("Recall:    %.2f%%\n",
        InformationTheoryMetrics.calculateRecall(actual, predicted) * 100);
    System.out.printf("F-Score:   %.2f%%\n",
        InformationTheoryMetrics.calculateFScore(actual, predicted) * 100);
  }
}