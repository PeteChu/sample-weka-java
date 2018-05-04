package com.company.SchecterZa;

import weka.classifiers.trees.J48;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.core.Instances;
import weka.classifiers.Evaluation;

public class Main {

    public static void main(String[] args) throws Exception {

        String train = "/Users/schecterza/Downloads/dataset/train.arff";
        String test = "/Users/schecterza/Downloads/dataset/test.arff";

        DataSource data = new DataSource(train);
        Instances dataraw = data.getDataSet();

        DataSource dataTest = new DataSource(test);
        Instances dataTestRaw = dataTest.getDataSet();

        StringToWordVector filter = new StringToWordVector();
        filter.setInputFormat(dataraw);

        Instances filteredData = filter.useFilter(dataraw, filter);
        Instances TestData = filter.useFilter(dataTestRaw, filter);

        J48 j48 = new J48();

        filteredData.setClassIndex(filteredData.attribute("@@class@@").index());
        TestData.setClassIndex(TestData.attribute("@@class@@").index());

        j48.buildClassifier(filteredData);

        Evaluation eval = new Evaluation(filteredData);
        eval.evaluateModel(j48, TestData);

        System.out.print(eval.toSummaryString());


    }
}
