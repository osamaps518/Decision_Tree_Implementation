package com.stegrandom.core;

public class Dataset {
  private String[][] x;
  private String[] y;

  public Dataset(String[][] x, String[] y) {
    this.x = x;
    this.y = y;
  }

  public String[][] getX() {
    return x;
  }

  public String[] getY() {
    return y;
  }

}
