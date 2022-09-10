import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Data2_Manager {

    // setting path of source data set
    protected static String inFile = "src/cross.pat";
    // lists that store all data sets
    protected static List<List<List<Double>>> training_dataSet = new LinkedList<>();
    protected static List<List<List<Double>>> training_desired = new LinkedList<>();
    protected static List<List<List<Double>>> testing_dataSet = new LinkedList<>();
    protected static List<List<List<Double>>> testing_desired = new LinkedList<>();
    protected static Data2_Manager data;

    static {
        try {
            data = new Data2_Manager();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Data2_Manager() throws IOException {

        for (int dataSet = 0; dataSet < 10; dataSet++) {
            FileReader fr = new FileReader(inFile);
            BufferedReader reader = new BufferedReader(fr);
            {
                // lists that store each data set
                List<List<Double>> sub_training_dataSet = new LinkedList<>();
                List<List<Double>> sub_training_desired = new LinkedList<>();
                List<List<Double>> sub_testing_dataSet = new LinkedList<>();
                List<List<Double>> sub_testing_desired = new LinkedList<>();

                // count lines
                int lines = 1;
                String data;

                /* splitting data into each type of list which are training dataset,
                   training desired output, testing dataset, and testing desired output
                 */
                while ((data = reader.readLine()) != null) {

                    if (lines % 3 == 0 || (lines + 1) % 3 == 0) {

                        String[] eachLine = data.split("\\s+");

                        List<Double> temp = new LinkedList<>();

                        for (String eachNum : eachLine) {
                            Double dataNum = Double.parseDouble(eachNum);
                            temp.add(dataNum);
                        }

                        if (lines % 3 == 0) {
                            if (lines % 10 == dataSet) {
                                sub_testing_desired.add(temp);
                            } else
                                sub_training_desired.add(temp);
                        } else if ((lines + 1) % 3 == 0) {
                            if ((lines + 1) % 10 == dataSet) {
                                sub_testing_dataSet.add(temp);
                            } else
                                sub_training_dataSet.add(temp);
                        }
                    }
                    lines++;
                }
                // insert a data set into a list of each type of data
                training_dataSet.add(sub_training_dataSet);
                training_desired.add(sub_training_desired);
                testing_dataSet.add(sub_testing_dataSet);
                testing_desired.add(sub_testing_desired);
            }
        }
    }

    public void setPath(String path) {
        inFile = path;
    }

    public static Data2_Manager getData() {
        return data;
    }

    /* these function below will be used for accessing data set the neuron network */
    public static List<List<Double>> getTrainData(int dataSet) {
        return training_dataSet.get(dataSet);
    }

    public static List<List<Double>> getTrainDs(int dataSet) {
        return training_desired.get(dataSet);
    }

    public static List<List<Double>> getTestData(int dataSet) {
        return testing_dataSet.get(dataSet);
    }

    public static List<List<Double>> getTestDs(int dataSet) {
        return testing_desired.get(dataSet);
    }

    public static void main(String[] args) {

        Data2_Manager.getData();

    }
}
