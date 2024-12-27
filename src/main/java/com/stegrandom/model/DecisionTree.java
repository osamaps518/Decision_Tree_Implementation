package com.stegrandom.model;

import com.stegrandom.core.Node;
import com.stegrandom.core.Dataset;

import java.util.*;

public class DecisionTree {
  private Node root;
  private int currentDepth;
  private int minSamplesAllowed;
  private int maxDepthAllowed;
  private double minEntropyDecreaseAllowed;

  public DecisionTree(Node root) {
    this.root = root;
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

  // public double predict(Object[] features) {
  // }

  public void train(Object[][] features, Object[] labels) {
  }

  public void buildTree(Object[][] x, Object[][] y) {

  }

  public void fit(String[][] x, String[] y, int n) {
    fit(root, x, y, n);
  }

  public void fit(Node node, String[][] x, String[] y, int depth) {
    // Base cases
    if (isPure(y) || depth >= maxDepthAllowed || x.length < minSamplesAllowed) {
      node.setPredictedClass(getMajorityClass(y));
      node.setDataPoints(new Dataset(x, y)); // Store data even in leaf nodes
      return;
    }

    // Find the best split
    int bestFeatureIndex = findBestSplit(x, y);
    double bestFeatureEntropy = Metrics.calculateEntropyAfterSplit(x, y, bestFeatureIndex);

    // Check if the split gives sufficient entropy decrease
    if (Metrics.calculateEntropy(x, y) - bestFeatureEntropy < minEntropyDecreaseAllowed) {
      node.setPredictedClass(getMajorityClass(y));
      node.setDataPoints(new Dataset(x, y));
      return;
    }

    // Store the current node's data before splitting
    node.setDataPoints(new Dataset(x, y));

    // Create children nodes
    node.setChildren(new HashMap<>());
    Set<String> uniqueValues = getUniqueValues(x, bestFeatureIndex);

    for (String value : uniqueValues) {
      Dataset split = splitData(x, y, bestFeatureIndex, value);
      if (split.getX().length > 0) {
        Node childNode = new Node();
        childNode.setLeafValue(value);
        childNode.setDataPoints(split); // Store the split result in the child node
        node.getChildren().put(value, childNode);

        // Recursive call
        fit(childNode, split.getX(), split.getY(), depth + 1);
      }
    }
  }

  private boolean isPure(String[] y) {
    String firstLabel = y[0];
    for (String label : y) {
      if (!label.equals(firstLabel))
        return false;
    }
    return true;
  }

  private String getMajorityClass(String[] y) {
    // Create a map to store how many times each class appears
    Map<String, Integer> counts = new HashMap<>();

    // Count occurrences of each class
    for (String label : y) {
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
    for (Map.Entry<String, Integer> entry : counts.entrySet()) {
      String className = entry.getKey();
      int classCount = entry.getValue();

      if (classCount > maxCount) {
        maxCount = classCount;
        majorityClass = className;
      }
    }

    return majorityClass;
  }

  private boolean shouldStopSplitting(String[][] x, String[] y) {
    if (x[0].length < minSamplesAllowed) {
      return true;
    }
    if (root.getChildren() == null) {
      return true;
    }
    if (currentDepth >= maxDepthAllowed) {
      return true;
    }
    if (Metrics.calculateEntropy(x, y) < minEntropyDecreaseAllowed) {
      return true;
    }
    return false;
  }

  // Split the data based on the unique values of the chosen feature
  public Dataset splitData(String[][] x, String[] y, int featureIndex, String featureValue) {
    List<String[]> newFeatures = new ArrayList<>();
    List<String> newTargets = new ArrayList<>();

    for (int i = 0; i < x[featureIndex].length; i++) {
      if (x[featureIndex][i].equals(featureValue)) {
        // Add both x and y to the node
        newFeatures.add(x[i]);
        newTargets.add(y[i]);
      }
    }

    // Convert lists to arrays
    String[][] newFeaturesArray = newFeatures.toArray(new String[0][]);
    String[] newTargetsArray = newTargets.toArray(new String[0]);

    // Create a SplitResult object to hold the split data
    return new Dataset(newFeaturesArray, newTargetsArray);
  }

  public int findBestSplit(String[][] x, String[] y) {
    // Initialize variables to track the best split
    double bestInformationGain = Double.NEGATIVE_INFINITY;
    int bestFeatureIndex = -1;

    // Iterate through each feature (column in x)
    for (int featureIndex = 0; featureIndex < x[0].length; featureIndex++) {
      // Extract the current feature column
      String[] featureColumn = new String[x.length];
      for (int row = 0; row < x.length; row++) {
        featureColumn[row] = x[row][featureIndex];
      }

      // Calculate information gain for this feature
      double currentIG = Metrics.calculateInformationGain(featureIndex, x, y);

      // Update best split if current is better
      if (currentIG > bestInformationGain) {
        bestInformationGain = currentIG;
        bestFeatureIndex = featureIndex;
      }
    }
    return bestFeatureIndex;
  }

}
