package com.stegrandom;

import java.util.*;

public class Main {
    public static void main(String[] args) {
        String[] array = { "value1", "value1", "value2", "value3" };
        Set<String> uniqueValues = new HashSet<>(Arrays.asList(array));
        System.out.println(Arrays.asList(uniqueValues.toString()));

        System.out.println("Hello world!");
    }
}