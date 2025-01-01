package com.stegrandom.model;

import com.stegrandom.core.Node;
import com.stegrandom.core.TrainingConfig;
import com.stegrandom.core.Dataset;

import java.util.*;

/**
 * A Decision Tree classifier implementation for categorical features.
 * This implementation uses information gain as the splitting criterion and
 * includes
 * various pre-pruning strategies to prevent overfitting.
 * 
 * 
 * The tree is built recursively, with each node representing a split on a
 * feature
 * that provides the maximum information gain. The building process continues
 * until
 * either a pure subset is achieved or one of the stopping criteria is met.
 * 
 *
 * 
 * Pre-pruning strategies include:
 * 
 * 1) Maximum depth limit
 * 2) 2) Minimum samples required for splitting
 * 3) Minimum entropy decrease threshold
 * 
 * 
 * 
 * 
 * @version 1.0
 */
public class DecisionTree {
  private Node root;
  private TrainingConfig config;
  private String[] featureNames;

  public DecisionTree() {
  }

  public Node getRoot() {
    return root;
  }

  public void setRoot(Node root) {
    this.root = root;
  }

  /**
   * Set feature names for better tree visualization
   * 
   * @param names Array of feature names
   */
  public void setFeatureNames(String[] names) {
    this.featureNames = names;
  }

  /**
   * Fits the decision tree to the training data.
   * This is the main method to train the decision tree classifier.
   * 
   * @param features a 2D array of String features where features[i][j] represents
   *                 the j-th feature of the i-th sample
   * @param target   an array of String labels corresponding to each sample
   * @param depth    the initial depth to start training (typically 0)
   * @throws IllegalArgumentException if input data is null or invalid
   */
  public void fit(String[][] features, String[] target, int depth) {
    checkNullValues(features, target);

    // Set the root node's dataset first
    root = new Node(new Dataset(features, target));

    // Calculate initial entropy once
    double initialEntropy = InformationTheoryMetrics.calculateEntropy(features, target);
    this.config = new TrainingConfig(initialEntropy, features.length);
    // Start the recursive process
    fit(root, features, target, depth);
  }

  /**
   * Validates the input data for null values and consistency.
   * 
   * @param features the feature matrix to validate
   * @param target   the target array to validate
   * @throws IllegalArgumentException if any validation fails
   */
  private void checkNullValues(String[][] features, String[] target) {
    // Validate input data
    if (features == null || target == null || features.length == 0 || target.length == 0) {
      throw new IllegalArgumentException("Features and target arrays cannot be null or empty");
    }

    if (features.length != target.length) {
      throw new IllegalArgumentException("Number of samples in features and target must match");
    }

    // Check for null values
    for (int i = 0; i < features.length; i++) {
      if (features[i] == null) {
        throw new IllegalArgumentException("Found null row in features at index " + i);
      }
      for (int j = 0; j < features[i].length; j++) {
        if (features[i][j] == null) {
          throw new IllegalArgumentException(
              String.format("Found null value in features at row %d, column %d", i, j));
        }
      }
      if (target[i] == null) {
        throw new IllegalArgumentException("Found null value in target at index " + i);
      }
    }
  }

  /**
   * Internal recursive method to build the decision tree.
   * This method handles the actual tree construction by recursively splitting
   * the data based on the feature that provides the maximum information gain.
   * 
   * @param node     the current Node in the tree being processed
   * @param features the feature matrix for the current split
   * @param target   the target array for the current split
   * @param depth    the current depth in the tree
   */
  public void fit(Node node, String[][] features, String[] target, int depth) {
    // Store the dataPoints in any case
    node.setDataPoints(new Dataset(features, target));

    // Find the best split
    int bestFeatureIndex = findBestSplit(features, target);

    if (shouldStopSplitting(features, target, bestFeatureIndex, depth)) {
      String majorityClass = getMajorityClass(target);
      node.setPredictedClass(majorityClass);
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

  /**
   * Determines whether to stop splitting based on various criteria.
   * 
   * @param features         the current feature matrix
   * @param target           the current target array
   * @param bestFeatureIndex the index of the best feature for splitting
   * @param depth            the current depth in the tree
   * @return true if splitting should stop, false otherwise
   */
  private boolean shouldStopSplitting(String[][] features, String[] target, int bestFeatureIndex, int depth) {
    // First, check for the pure subset case
    if (isPure(target)) {
      return true;
    }

    // Check depth and sample size against pre-configured limits
    // These limits are calcualted in the TrainingConfig class
    if (depth >= config.getMaxDepthAllowed() ||
        features.length < config.getMinSamplesAllowed()) {
      return true;
    }

    // Calculate entropy decrease to see if this split is worthwhile
    double currentEntropy = InformationTheoryMetrics.calculateEntropy(features, target);
    double entropyAfterSplit = InformationTheoryMetrics.calculateEntropyAfterSplit(features, target, bestFeatureIndex);
    double entropyDecrease = currentEntropy - entropyAfterSplit;

    // Compare against our minimum entropy decrease threshold
    // If the decrease is too small, it's not worth making this split
    if (entropyDecrease < config.getMinEntropyDecreaseAllowed()) {
      return true;
    }

    // If all stopping conditions are passed, we should continue splitting
    return false;
  }

  /**
   * Gets unique values from a specific feature column.
   * 
   * @param features     the feature matrix
   * @param featureIndex the index of the feature to get unique values from
   * @return a Set of unique String values for the specified feature
   */
  private Set<String> getUniqueValues(String[][] features, int featureIndex) {
    Set<String> uniqueValues = new HashSet<>();
    for (String[] row : features) {
      uniqueValues.add(row[featureIndex]);
    }
    return uniqueValues;
  }

  /**
   * Checks if a target array contains only one unique class (is pure).
   * 
   * @param target the target array to check
   * @return true if the target array is pure, false otherwise
   */
  private boolean isPure(String[] target) {
    String firstLabel = target[0];
    for (String label : target) {
      if (!label.equals(firstLabel))
        return false;
    }
    return true;
  }

  /**
   * Determines the majority class in a target array.
   * 
   * @param target the target array to analyze
   * @return the most frequent class label in the target array
   */
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

  /**
   * Splits the dataset based on a specific feature value.
   * 
   * @param features     the feature matrix to split
   * @param targets      the target array to split
   * @param featureIndex the index of the feature to split on
   * @param featureValue the value of the feature to split on
   * @return a Dataset containing the split data
   */
  public Dataset splitData(String[][] features, String[] targets, int featureIndex, String featureValue) {
    List<String[]> newFeatures = new ArrayList<>();
    List<String> newTargets = new ArrayList<>();

    for (int i = 0; i < features.length; i++) {
      String[] row = features[i];
      if (row[featureIndex].equals(featureValue)) {
        // Add both features and target values to the new split
        String[] newRow = Arrays.copyOf(row, row.length);
        newFeatures.add(newRow);
        newTargets.add(targets[i]);
      }
    }

    return new Dataset(newFeatures.toArray(new String[0][]),
        newTargets.toArray(new String[0]));
  }

  /**
   * Finds the best feature to split on based on information gain.
   * 
   * @param features the feature matrix
   * @param target   the target array
   * @return the index of the feature that provides the highest information gain
   */
  public int findBestSplit(String[][] features, String[] target) {
    double bestInformationGain = Double.NEGATIVE_INFINITY;
    int bestFeatureIndex = -1;

    // Calculate entropy of entire dataset before any splits
    double baseEntropy = InformationTheoryMetrics.calculateEntropy(features, target);

    // Evaluate each feature as a potential split point
    for (int featureIndex = 0; featureIndex < features[0].length; featureIndex++) {
      // Find all unique values for this feature
      Set<String> uniqueValues = new HashSet<>();
      for (int row = 0; row < features.length; row++) {
        uniqueValues.add(features[row][featureIndex]);
      }

      // Calculate weighted entropy across all values of this feature
      double weightedEntropy = 0;
      for (String value : uniqueValues) {
        // For each unique value, gather target values that match this feature value
        int totalCount = 0;
        List<String> subtargets = new ArrayList<>();

        for (int row = 0; row < features.length; row++) {
          if (features[row][featureIndex].equals(value)) {
            totalCount++;
            subtargets.add(target[row]);
          }
        }

        // Calculate probability of this value occurring
        double probability = (double) totalCount / features.length;

        // Calculate entropy of the subset with this feature value
        double subEntropy = InformationTheoryMetrics.calculateEntropy(
            new String[0][0], // Empty features array since we only need target values
            subtargets.toArray(new String[0]));

        // Add weighted entropy contribution from this subset
        weightedEntropy += probability * subEntropy;
      }

      // Information gain is reduction in entropy after split
      double informationGain = baseEntropy - weightedEntropy;

      // Keep track of feature that gives best information gain
      if (informationGain > bestInformationGain) {
        bestInformationGain = informationGain;
        bestFeatureIndex = featureIndex;
      }
    }

    return bestFeatureIndex;
  }

  /**
   * Makes predictions for multiple samples.
   * 
   * @param testData a 2D array of features to make predictions for
   * @return an array of predicted class labels
   */
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

  /**
   * Makes a prediction for a single sample.
   * 
   * @param row the feature array to make a prediction for
   * @return the predicted class label
   */
  private String predict(String[] row) {
    Node currentNode = root;
    // Keep track of the last valid node with a prediction
    Node lastValidNode = root;

    while (!currentNode.isLeaf()) {
      int featureIdx = currentNode.getSplitFeatureIndex();
      // Get the feature value from the test row using the node's split feature
      String featureValue = row[currentNode.getSplitFeatureIndex()];

      Node nextNode = currentNode.getNextNode(featureValue);
      // If can't find a matching child node
      if (nextNode == null) {

        // Before breaking, ensure current node has a prediction
        if (currentNode.getPredictedClass() == null) {
          // Use the training data at this node to make a prediction
          Dataset nodeData = currentNode.getDataPoints();
          if (nodeData != null && nodeData.getY() != null && nodeData.getY().length > 0) {
            currentNode.setPredictedClass(getMajorityClass(nodeData.getY()));
          } else {
            // If no data available at current node, use last valid node's prediction
            currentNode.setPredictedClass(lastValidNode.getPredictedClass());
          }
        }
        break;
      }

      // Before moving to next node, update last valid node if current has a
      // prediction
      if (currentNode.getPredictedClass() != null) {
        lastValidNode = currentNode;
      }

      currentNode = nextNode;
    }

    // Final safety check - if still don't have a prediction,
    // use the root node's majority class
    if (currentNode.getPredictedClass() == null) {
      Dataset rootData = root.getDataPoints();
      String rootPrediction = getMajorityClass(rootData.getY());
      currentNode.setPredictedClass(rootPrediction);
    }

    return currentNode.getPredictedClass();
  }

  /**
   * Public method to start printing the tree
   */
  public void printTree() {
    if (root == null) {
      System.out.println("Tree is empty");
      return;
    }
    System.out.println("\nDecision Tree Structure:");
    System.out.println("========================");
    printNode(root, "", "ROOT");
  }

  /**
   * Private recursive method to print each node
   * 
   * @param node        Current node being printed
   * @param indent      Current indentation string
   * @param branchLabel Label for the current branch
   */
  private void printNode(Node node, String indent, String branchLabel) {
    // Print current node with its branch label
    System.out.print(indent + "├── " + branchLabel);

    if (node.isLeaf()) {
      // For leaf nodes, print the prediction
      System.out.println(" → " + node.getPredictedClass());
    } else {
      // For internal nodes, print the split feature name or index
      int featureIndex = node.getSplitFeatureIndex();
      String featureName = featureNames != null && featureIndex < featureNames.length ? featureNames[featureIndex]
          : "feature " + featureIndex;
      System.out.println(" [Split on " + featureName + "]");

      // Print each child with increased indentation
      String newIndent = indent + "│   ";
      Map<String, Node> children = node.getChildren();

      // Sort the feature values for consistent output
      List<String> sortedFeatureValues = new ArrayList<>(children.keySet());
      Collections.sort(sortedFeatureValues);

      for (String featureValue : sortedFeatureValues) {
        Node child = children.get(featureValue);
        printNode(child, newIndent, featureValue);
      }
    }
  }
}