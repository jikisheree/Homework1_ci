import java.util.Random;

public class WeightGraph {
    protected int row, column;
    protected Double[][] weightData;

    public WeightGraph(int row, int column, boolean ifRandom) {
        this.row = row;
        this.column = column;

        this.weightData = new Double[row][column];

        Random rand = new Random();

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (ifRandom)
                    weightData[i][j] = rand.nextDouble(-1,1);
                else
                    weightData[i][j] = 0.0;
            }
        }

    }

    void setWeight(int row, int column, Double weight){
        weightData[row][column] = weight;
    }

    Double getWeight(int row, int column){return weightData[row][column];}

}
