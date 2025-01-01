package com.stegrandom;

import java.util.*;

import com.stegrandom.model.DecisionTree;
import com.stegrandom.model.InformationTheoryMetrics;

public class Main {
    public static void main(String[] args) {
        // String[] array = { "value1", "value1", "value2", "value3" };
        // Set<String> uniqueValues = new HashSet<>(Arrays.asList(array));
        // System.out.println(Arrays.asList(uniqueValues.toString()));
        //
        // System.out.println("Hello world!");

        // Example data
        // String[][] x = {
        // { "sunny", "hot", "high", "false" },
        // { "sunny", "hot", "high", "true" },
        // { "overcast", "hot", "high", "false" },
        // { "rainy", "mild", "high", "false" },
        // { "rainy", "cool", "normal", "false" }
        // };
        // String[] y = { "no", "no", "yes", "yes", "yes" };

        // double entropy = InformationTheoryMetrics.calculateEntropy(x, y);
        // System.out.println("Entropy: " + entropy);

        // double informationGain = InformationTheoryMetrics.calculateInformationGain(0,
        // x, y);
        // System.out.println("Information Gain: " + informationGain);

        // double accuracy = InformationTheoryMetrics.calculateAccuracy(new double[] {
        // 1, 0, 1, 1 },
        // new double[] { 1, 0, 0, 1 });
        // System.out.println("Accuracy: " + accuracy);

        // double precision = InformationTheoryMetrics.calculatePrecision(new double[] {
        // 1, 0, 1, 1 },
        // new double[] { 1, 0, 0, 1 });
        // System.out.println("Precision: " + precision);

        // double recall = InformationTheoryMetrics.calculateRecall(new double[] { 1, 0,
        // 1, 1 },
        // new double[] { 1, 0, 0, 1 });
        // System.out.println("Recall: " + recall);

        // double fScore = InformationTheoryMetrics.calculateFScore(new double[] { 1, 0,
        // 1, 1 },
        // new double[] { 1, 0, 0, 1 });
        // System.out.println("F-Score: " + fScore);
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

        double accuracy = InformationTheoryMetrics.calculateAccuracy(actualValuesDouble,
                predictedValuesDouble);
        double precision = InformationTheoryMetrics.calculatePrecision(actualValuesDouble,
                predictedValuesDouble);
        double recall = InformationTheoryMetrics.calculateRecall(actualValuesDouble,
                predictedValuesDouble);
        double fscore = InformationTheoryMetrics.calculateFScore(actualValuesDouble,
                predictedValuesDouble);

        System.out.println("Model Performance:");
        System.out.println("Accuracy: " + accuracy);
        System.out.println("Precision: " + precision);
        System.out.println("Recall: " + recall);
        System.out.println("F-Score: " + fscore);
    }
}