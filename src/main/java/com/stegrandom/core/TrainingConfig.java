package com.stegrandom.core;

public class TrainingConfig {
  private final int minSamplesAllowed;
  private final int maxDepthAllowed;
  private final double minEntropyDecreaseAllowed;

  public TrainingConfig(double initialEntropy, int n) {
    this.minSamplesAllowed = (int) Math.sqrt(n) / 10;
    this.maxDepthAllowed = (int) Math.log(n);
    this.minEntropyDecreaseAllowed = initialEntropy / 100;
  }

  public int getMinSamplesAllowed() {
    return minSamplesAllowed;
  }

  public int getMaxDepthAllowed() {
    return maxDepthAllowed;
  }

  public double getMinEntropyDecreaseAllowed() {
    return minEntropyDecreaseAllowed;
  }
}
