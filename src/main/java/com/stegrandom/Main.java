package com.stegrandom;

import java.util.*;

import com.stegrandom.model.DecisionTree;

public class Main {
    public static void main(String[] args) {
        String[] array = { "value1", "value1", "value2", "value3" };
        Set<String> uniqueValues = new HashSet<>(Arrays.asList(array));
        System.out.println(Arrays.asList(uniqueValues.toString()));

        System.out.println("Hello world!");
    }

    public void testModel(String[][] trainX, String[] trainY,
            String[][] testX, String[] testY) {
        // Train the model
        DecisionTree tree = new DecisionTree();
        tree.fit(trainX, trainY, 5);

        // Make predictions on test data
        String[] predictions = tree.predict(testX);

        // Convert String arrays to double arrays for metrics
        double[] actualValuesDouble = Arrays.stream(testY)
                .mapToDouble(label -> label.equals("positive") ? 1.0 : 0.0)
                .toArray();
        double[] predictedValuesDouble = Arrays.stream(predictions)
                .mapToDouble(label -> label.equals("positive") ? 1.0 : 0.0)
                .toArray();

        // Calculate and display all metrics
        Metrics metrics = new MetricsImplementation(); // Motyam implementation

        double accuracy = metrics.calculateAccuracy(actualValuesDouble, predictedValuesDouble);
        double precision = metrics.calculatePrecision(actualValuesDouble, predictedValuesDouble);
        double recall = metrics.calculateRecall(actualValuesDouble, predictedValuesDouble);
        double fscore = metrics.calculateFScore(actualValuesDouble, predictedValuesDouble);

        System.out.println("Model Performance:");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F-Score: " + fscore);
    }
}