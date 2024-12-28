package com.stegrandom.core;

import java.util.*;

public class Node {
    private Dataset dataPoints; // The data points that reach this node
    private String predictedClass;
    Map<String, Node> children; // empty if leaf
    private Integer splitFeatureIndex;

    public Node(Dataset dataPoints) {
        this.dataPoints = dataPoints;
        this.children = new HashMap<>();
    }

    public void setSplitFeatureIndex(int featureIndex) {
        this.splitFeatureIndex = featureIndex;
    }

    public Dataset getDataPoints() {
        return dataPoints;
    }

    public void setDataPoints(Dataset splitResult) {
        this.dataPoints = splitResult;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public boolean isLeaf() {
        return children.isEmpty(); // A leaf node has no children
    }

    public Map<String, Node> getChildren() {
        return children;
    }

    public void setChildren(Map<String, Node> children) {
        this.children = children;
    }

    // During prediction
    public Node getNextNode(String featureValue) {
        return children != null ? children.get(featureValue) : null;
    }
}
