package com.stegrandom.core;

import java.util.*;

public class Node {
    private Dataset dataPoints; // The data points that reach this node
    private String predictedClass;
    Map<String, Node> children; // null if leaf
    private String leafValue; // The decision/value at this node

    public Node() {

    }

    public Node(Dataset dataPoints, String predictedClass, String leafValue) {
        this.dataPoints = dataPoints;
        this.predictedClass = predictedClass;
        this.leafValue = leafValue;
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

    public String getLeafValue() {
        return leafValue;
    }

    public void setLeafValue(String leafValue) {
        this.leafValue = leafValue;
    }

    public boolean isLeaf() {
        return children == null;
    }

    public Map<String, Node> getChildren() {
        return children;
    }

    public void setChildren(Map<String, Node> children) {
        this.children = children;
    }
}
