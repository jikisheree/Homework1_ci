import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class Data1_Manager {

    protected static String inFile = "src/Flood_dataset.txt";
    protected static List<List<List<Double>>> training_dataSet = new ArrayList<>();
    protected static List<List<List<Double>>> training_desired = new ArrayList<>();
    protected static List<List<List<Double>>> testing_dataSet = new ArrayList<>();
    protected static List<List<List<Double>>> testing_desired = new ArrayList<>();
    protected static Data1_Manager data;

    static {
        try {
            data = new Data1_Manager();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Data1_Manager() throws IOException {

             for (int dataSet=0; dataSet<10; dataSet++) {
                 FileReader fr = new FileReader(inFile);
                 BufferedReader reader = new BufferedReader(fr);
                 {

                     List<List<Double>> sub_training_dataSet = new ArrayList<>();
                     List<List<Double>> sub_training_desired = new ArrayList<>();
                     List<List<Double>> sub_testing_dataSet = new ArrayList<>();
                     List<List<Double>> sub_testing_desired = new ArrayList<>();

                     int lines = 0;
                     String data;

                     // checking each line
                     while ((data = reader.readLine()) != null) {
                         lines++;

                         String[] eachLine = data.split("\t");

                         List<Double> Input_line = new LinkedList<>();
                         List<Double> Desired_line = new LinkedList<>();
                         for (String eachNum : eachLine) {
                             Double dataNum = Double.parseDouble(eachNum) / 700;
                             Input_line.add(dataNum);
                         }
                         Desired_line.add(Input_line.get(Input_line.size() - 1));
                         Input_line.remove(Input_line.size() - 1);

                         if (lines % (10) == dataSet) {
                             sub_testing_dataSet.add(Input_line);
                             sub_testing_desired.add(Desired_line);
                         } else {
                             sub_training_dataSet.add(Input_line);
                             sub_training_desired.add(Desired_line);
                         }
                     }

                     // insert data into sets
                     training_dataSet.add(sub_training_dataSet);
                     training_desired.add(sub_training_desired);
                     testing_dataSet.add(sub_testing_dataSet);
                     testing_desired.add(sub_testing_desired);
                 }

             }
    }

    public void setPath(String path){
        inFile = path;
    }

    public static Data1_Manager getData(){
        return data;}

    public static List<List<Double>> getTrainData(int dataSet){
        return training_dataSet.get(dataSet);
    }

    public static List<List<Double>> getTrainDs(int dataSet){
        return training_desired.get(dataSet);
    }

    public static List<List<Double>> getTestData(int dataSet){
        return testing_dataSet.get(dataSet);
    }

    public static List<List<Double>> getTestDs(int dataSet){
        return testing_desired.get(dataSet);
    }

    public static void main(String[] args) throws IOException {

        Data1_Manager.getData();

    }
}


