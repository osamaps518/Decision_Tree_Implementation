package com.stegrandom.core;

public class Feature {
    private Object value;        // Can hold either String (categorical) or Double (continuous)
    private FeatureType type;    // The type of this feature (CATEGORICAL or CONTINUOUS)

    public Feature(Object value, FeatureType type) {
        this.value = value;
        this.type = type;
    }

    public Object getValue() {
        return value;
    }

    public void setValue(Object value) {
        this.value = value;
    }

    public FeatureType getType() {
        return type;
    }

    public void setType(FeatureType type) {
        this.type = type;
    }
}
