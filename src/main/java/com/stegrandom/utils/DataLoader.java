package com.stegrandom.utils;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DataLoader {
    private String filePath;

    public DataLoader(String filePath) {
        this.filePath = filePath;
    }

    public Object[][] load() throws IOException {
        Object[][] features = null;
        // Create a buffered reader to efficiently read the file
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            // Read the header line first - it contains feature names
            String headerLine = reader.readLine();
            if (headerLine != null) {
                // Split the header line to get feature names
                String[] headers = headerLine.split(",");

                // Create a list to store all data rows
                List<Object[]> dataRows = new ArrayList<>();

                // Read the rest of the file line by line
                String line;
                while ((line = reader.readLine()) != null) {
                    // Split each line into an array of values
                    String[] values = line.split(",");
                    Object[] typedValues = new Object[values.length];

                    for (int i = 0; i < values.length; i++) {
                        typedValues[i] = parseValue(values[i]);
                    }

                    dataRows.add(typedValues);
                }

                // Convert list to array for easier processing
                features = dataRows.toArray(new Object[0][]);
            }
        }
        return features;
    }

    private static Object parseValue(String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e1) {
            try {
                return Double.parseDouble(value);
            } catch (NumberFormatException e2) {
                return value;
            }
        }
    }
}
