package com.stegrandom.model;

import com.stegrandom.core.Node;
import com.stegrandom.core.Dataset;

import java.util.*;

public class DecisionTree {
  private Node root;
  private int minSamplesAllowed;
  private int maxDepthAllowed;
  private double minEntropyDecreaseAllowed;

  public DecisionTree() {
    this.currentDepth = 0;
  }

  public Node getRoot() {
    return root;
  }

  public void setRoot(Node root) {
    this.root = root;
  }

  public void decidePrePruningDefaults(double initialEntropy, int n) {
    // Decide minSampels as the square root of the number of samples/10
    // Decide maxDepth as the log2 of the number of features
    // if initial entropy (at the root node) is E, then setting min_entropy_decrease
    // to E/100

    minSamplesAllowed = (int) Math.sqrt(n) / 10;
    maxDepthAllowed = (int) Math.log(n);
    minEntropyDecreaseAllowed = (initialEntropy / 100);
  }

  public void train(Object[][] features, Object[] labels) {
  }

  public void buildTree(Object[][] x, Object[][] y) {

  }

  public String[] predict(String[][] testData) {
    // Create an array to hold predictions for each row
    String[] predictions = new String[testData.length];

    // For each row in our test data
    for (int i = 0; i < testData.length; i++) {
      // Get prediction for this single row using our private predict method
      predictions[i] = predict(testData[i]);
    }

    return predictions;
  }

  private String predict(String[] row) {
    // Start at the root of the decision tree
    Node currentNode = root;

    // Keep moving down the tree until hitting a leaf
    while (!currentNode.isLeaf()) {
      // Get the feature value from the test row using the node's split feature
      String featureValue = row[currentNode.getSplitFeatureIndex()];
      Node nextNode = currentNode.getNextNode(featureValue);

      // If can't find a matching child node, stop here and use current node's
      // majority class
      if (nextNode == null) {
        break;
      }

      // Use that value to move to the next node
      currentNode = nextNode;
    }

    // Once a leaf is reached, return its prediction
    return currentNode.getPredictedClass();
  }

  public void fit(String[][] features, String[] target, int depth) {
    fit(root, features, target, depth);
  }

  public void fit(Node node, String[][] features, String[] target, int depth) {
    // Store the dataPoints in any case
    node.setDataPoints(new Dataset(features, target));
    // Find the best split
    int bestFeatureIndex = findBestSplit(features, target);

    if (shouldStopSplitting(features, target, bestFeatureIndex, depth)) {
      node.setPredictedClass(getMajorityClass(target));
      return;
    }

    // Get unique values of the best feature
    Set<String> uniqueValues = getUniqueValues(features, bestFeatureIndex);
    node.setSplitFeatureIndex(bestFeatureIndex);

    for (String value : uniqueValues) {
      Dataset split = splitData(features, target, bestFeatureIndex, value);
      if (split.getX().length > 0) {
        Node childNode = new Node(split);
        node.getChildren().put(value, childNode);

        // Recursive call to continue growing the tree
        fit(childNode, split.getX(), split.getY(), depth + 1);
      }
    }
  }

  // Checks if there's full purety or any of the prepruning conditions is
  // activated
  private boolean shouldStopSplitting(String[][] features, String[] target, int bestFeatureIndex, int depth) {
    // Base cases
    if (isPure(target) || depth >= maxDepthAllowed || features.length < minSamplesAllowed)
      return true;

    double bestFeatureEntropy = Metrics.calculateEntropyAfterSplit(features, target, bestFeatureIndex);

    // Check if the split gives sufficient entropy decrease
    if (Metrics.calculateEntropy(features, target) - bestFeatureEntropy < minEntropyDecreaseAllowed)
      return true;

    return false;
  }

  private Set<String> getUniqueValues(String[][] features, int featureIndex) {
    // A set is only allowed to have unique values, any redundant values will be
    // discarded
    Set<String> uniqueValues = new HashSet<>(Arrays.asList(features[featureIndex]));
    return uniqueValues;
  }

  private boolean isPure(String[] target) {
    String firstLabel = target[0];
    for (String label : target) {
      if (!label.equals(firstLabel))
        return false;
    }
    return true;
  }

  private String getMajorityClass(String[] target) {
    // Create a map to store how many times each class appears
    Map<String, Integer> counts = new HashMap<>();

    // Count occurrences of each class
    for (String label : target) {
      // If we've seen this label before, get its current count
      // If we haven't seen it, use 0 as the starting count
      int currentCount = counts.getOrDefault(label, 0);

      // Add 1 to the count and update the map
      counts.put(label, currentCount + 1);
    }

    // Find the class with the highest count
    String majorityClass = null;
    int maxCount = 0;

    // Go through each class and its count
    for (var entry : counts.entrySet()) {
      String className = entry.getKey();
      int classCount = entry.getValue();

      if (classCount > maxCount) {
        maxCount = classCount;
        majorityClass = className;
      }
    }

    return majorityClass;
  }

  // Split the data based on the unique values of the chosen feature
  public Dataset splitData(String[][] features, String[] targets, int featureIndex, String featureValue) {
    List<String[]> newFeatures = new ArrayList<>();
    List<String> newTargets = new ArrayList<>();

    for (int i = 0; i < features[featureIndex].length; i++) {
      if (features[featureIndex][i].equals(featureValue)) {
        // Add both x and y to the node
        newFeatures.add(features[i]);
        newTargets.add(targets[i]);
      }
    }

    // Convert lists to arrays
    String[][] newFeaturesArray = newFeatures.toArray(new String[0][]);
    String[] newTargetsArray = newTargets.toArray(new String[0]);

    // Create a SplitResult object to hold the split data
    return new Dataset(newFeaturesArray, newTargetsArray);
  }

  public int findBestSplit(String[][] features, String[] target) {
    // Initialize variables to track the best split
    double bestInformationGain = Double.NEGATIVE_INFINITY;
    int bestFeatureIndex = -1;

    // Iterate through each feature (column in x)
    for (int featureIndex = 0; featureIndex < features[0].length; featureIndex++) {
      // Extract the current feature column
      String[] featureColumn = new String[features.length];
      for (int row = 0; row < features.length; row++) {
        featureColumn[row] = features[row][featureIndex];
      }

      // Calculate information gain for this feature
      double currentIG = Metrics.calculateInformationGain(featureIndex, features, target);

      // Update best split if current is better
      if (currentIG > bestInformationGain) {
        bestInformationGain = currentIG;
        bestFeatureIndex = featureIndex;
      }
    }
    return bestFeatureIndex;
  }

}
