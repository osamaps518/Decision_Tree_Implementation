package com.stegrandom.core;

import java.util.List;

public class Node {
    private List<Double> dataPoints;    // The data points that reach this node
    private String predictedClass;
    private Node leftNode;
    private Node rightNode;
    private String leafValue;           // The decision/value at this node

    public Node(List<Double> dataPoints, String predictedClass, Node leftNode, Node rightNode, String leafValue) {
        this.dataPoints = dataPoints;
        this.predictedClass = predictedClass;
        this.leftNode = leftNode;
        this.rightNode = rightNode;
        this.leafValue = leafValue;
    }

    public List<Double> getDataPoints() {
        return dataPoints;
    }

    public void setDataPoints(List<Double> dataPoints) {
        this.dataPoints = dataPoints;
    }

    public String getPredictedClass() {
        return predictedClass;
    }

    public void setPredictedClass(String predictedClass) {
        this.predictedClass = predictedClass;
    }

    public Node getRightNode() {
        return rightNode;
    }

    public void setRightNode(Node rightNode) {
        this.rightNode = rightNode;
    }

    public String getLeafValue() {
        return leafValue;
    }

    public void setLeafValue(String leafValue) {
        this.leafValue = leafValue;
    }

    public Node getLeftNode() {
        return leftNode;
    }

    public void setLeftNode(Node leftNode) {
        this.leftNode = leftNode;
    }

    public boolean isLeaf() {
        return leafValue != null;
    }
}
