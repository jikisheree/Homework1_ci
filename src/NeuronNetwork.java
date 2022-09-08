import java.util.List;
import java.util.Random;

public class NeuronNetwork {

    protected Double minError;
    protected int maxEpoch;
    protected Double bias;
    protected Double learningRate;
    protected Double mmRate;
    private final List<List<Double>> training_dataSet;
    private final List<List<Double>> training_desired;
    private final List<List<Double>> testing_dataSet;
    private final List<List<Double>> testing_desired;
    private Double[][] node, local_gradient;
    private int[] nodeLayer;
    private WeightGraph[] weightOfLayer;
    private WeightGraph[] changedWeight;
    private Double[] error;
    private int weightLayerNum;
    private int nodeLayerNum;
    private final int dataSet;

    public NeuronNetwork(Double minError, Double learningRate, Double mm, int maxEpoch, Double bias,
                         int[] hidden, int dataSet) {

        this.minError = minError;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.bias = bias;
        this.mmRate = mm;
        this.dataSet = dataSet;

        this.training_dataSet = Data1_Manager.getTrainData(this.dataSet);
        this.training_desired = Data1_Manager.getTrainDs(this.dataSet);
        this.testing_dataSet = Data1_Manager.getTestData(this.dataSet);
        this.testing_desired = Data1_Manager.getTestDs(this.dataSet);

        nodeBuilding(hidden);

    }

    public void nodeBuilding(int[] hidden) {

        int hNum = hidden.length;
        this.nodeLayer = new int[hNum + 2];
        // add input nodes
        this.nodeLayer[0] = training_dataSet.get(dataSet).size();
        for (int i = 1; i <= hNum + 1; i++) {

            if (i == hNum + 1) {
                // add output nodes
                this.nodeLayer[i] = training_desired.get(dataSet).size();
                this.error = new Double[this.nodeLayer[i]];
            } else
                // add hidden nodes
                this.nodeLayer[i] = hidden[i - 1];

        }
        // store total number of layers
        this.nodeLayerNum = nodeLayer.length;
        // store total number of line between layer (weight)
        this.weightLayerNum = nodeLayerNum-1;
        // matrix to store value of each node in each layer
        this.node = new Double[nodeLayerNum][];
        //// matrix to store local gradient of each node in each layer
        this.local_gradient = new Double[nodeLayerNum][];

        for (int i = 0; i < nodeLayerNum; i++) {
            this.node[i] = new Double[nodeLayer[i]];
            this.local_gradient[i] = new Double[nodeLayer[i]];
        }

        // matrix of overall neuron network
        this.weightOfLayer = new WeightGraph[weightLayerNum];
        this.changedWeight = new WeightGraph[weightLayerNum];

        for (int i = 0; i < weightLayerNum; i++) {
            weightOfLayer[i] = new WeightGraph(nodeLayer[i + 1], nodeLayer[i], true);
            changedWeight[i] = new WeightGraph(nodeLayer[i + 1], nodeLayer[i], false);
        }

    }

    public void training() {

        System.out.println("================= TRAINING " + (dataSet + 1) + " =================");
        int n = 0;
        double avgError = 10000.0;
        int inNodeNum = nodeLayer[0];
        Random ranDataLine = new Random();
        while (n < maxEpoch && avgError > minError) {
            double sum_error = 0.0;
            // insert input value to node
            for (int l = 0; l < training_dataSet.size(); l++) {
                int lineNum = ranDataLine.nextInt(training_dataSet.size());

                for (int i = 0; i < inNodeNum; i++) {
                    this.node[0][i] = training_dataSet.get(lineNum).get(i);
                }

                feedForward();
                errorCalculation(lineNum, true);
                backPropagation();

                double d = training_desired.get(lineNum).get(0) * 700;
                double g = node[nodeLayerNum - 1][0] * 700;
//                System.out.println("desired:" + (int)d + " get: "+ g + "\t error_n: " + Math.abs(d-g));

                sum_error += 0.5 * Math.pow(error[0], 2);
            }
            avgError = sum_error / training_dataSet.size();
//            System.out.println("N epoch: "+n +"\t" + avgError);
            n++;
        }

        System.out.println("final Average error: " + avgError);
    }

    public void testing() {

        System.out.println("================= TESTING =================");
        double avgError;
        int inNodeNum = nodeLayer[0];
        double sum_error = 0.0;

        for (int l = 0; l < Data1_Manager.testing_dataSet.size(); l++) {
            // insert input value to node
            for (int i = 0; i < inNodeNum; i++) {
                this.node[0][i] = Data1_Manager.getTestData(dataSet).get(l).get(i);
            }

            feedForward();
            errorCalculation(l, false);

            double d = Data1_Manager.getTestDs(dataSet).get(l).get(0) * 700;
            double g = node[nodeLayerNum - 1][0] * 700;
            System.out.println("desired:" + (int) d + " get: " + g + "\t error_n: " + Math.abs(d - g));

            sum_error += 0.5 * Math.pow(error[0], 2);
        }
        avgError = sum_error / Data1_Manager.testing_dataSet.size();

        System.out.println("Average error: " + avgError);
    }

    public void feedForward() {

        for (int i = 0; i < weightLayerNum; i++) {
            // number of node in each layer
            int nodeAfterNum = nodeLayer[i + 1];
            int nodeBeforeNum = nodeLayer[i];

            // node in previous layer will be row and next layer will be column
            for (int row = 0; row < nodeAfterNum; row++) {
                double sum = 0.0;
                for (int col = 0; col < nodeBeforeNum; col++) {
                    // weight of each line * value of node before line
                    Double weightOfLine = this.weightOfLayer[i].getWeight(row, col);
                    sum += (weightOfLine * activation(node[i][col]));
                }
                node[i + 1][row] = sum + bias;
            }
        }
    }

    public void errorCalculation(int lineNum, boolean ifTrain) {

        Double desired;
        // nodeLayerNUm-1 = index of node layer output
        for (int i = 0; i < nodeLayer[nodeLayerNum - 1]; i++) {
            if (ifTrain)
                desired = training_desired.get(lineNum).get(i);
            else
                desired = testing_desired.get(lineNum).get(i);
            this.error[i] = desired - node[nodeLayerNum - 1][i];
            // activation function of output layer is linear
            this.local_gradient[nodeLayerNum - 1][i] = this.error[i];
        }
    }

    private void backPropagation() {

        // finding local gradient in hidden layer
        //finding delta weight in hidden layer
        // changing all weight in all layer

        // each layer form backward
        for (int layer = weightLayerNum - 1; layer >= 0; layer--) {
            // each node in layer before weight line
            for (int j = 0; j < nodeLayer[layer]; j++) {
                double sum = 0.0;
                // each node in layer after weight line
                for (int k = 0; k < nodeLayer[layer + 1]; k++) {
                    sum += local_gradient[layer + 1][k] * weightOfLayer[layer].getWeight(k, j);
                }
                local_gradient[layer][j] = activation_diff(node[layer][j]) * sum;
                for (int k = 0; k < nodeLayer[layer + 1]; k++) {

                    Double cw = (mmRate * changedWeight[layer].getWeight(k, j)) +
                            (learningRate * local_gradient[layer + 1][k] * activation(node[layer][j]));

                    changedWeight[layer].setWeight(k, j, cw);

                    weightOfLayer[layer].setWeight(k, j,
                            (weightOfLayer[layer].getWeight(k, j) + changedWeight[layer].getWeight(k, j)));
                }
            }
        }
    }

    public double activation(Double value) {
        // tanh
        return (Math.exp(value) - Math.exp(-value)) / (Math.exp(value) + Math.exp(-value));
        // relu
//        return Math.max(0.01, value);
    }

    public double activation_diff(Double value) {
        // tanh
        return 1.0 - Math.pow(activation(value), 2);
//        if(value<=0) return  0.01;
//        else return 1;
    }


}
