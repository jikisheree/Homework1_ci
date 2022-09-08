import java.util.List;
import java.util.Random;

public class NeuronNetwork2 {

    protected Double minError;
    protected int maxEpoch;
    protected Double bias;
    protected Double learningRate;
    protected Double mmRate;
    private List<List<Double>> training_dataSet, training_desired, testing_dataSet, testing_desired;
    private Double[][] node, local_gradient;
    private int[] nodeLayer;
    private WeightGraph[] weightOfLayer;
    private WeightGraph[] changedWeight;
    private Double[] error;
    private int weightLayerNum;
    private int nodeLayerNum;
    private int dataSet;

    public NeuronNetwork2(Double minError, Double learningRate, Double mm, int maxEpoch, Double bias,
                          int[] hidden, int dataSet) {

        this.minError = minError;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.bias = bias;
        this.mmRate = mm;
        this.dataSet = dataSet;

        this.training_dataSet = Data2_Manager.getTrainData(this.dataSet);
        this.training_desired = Data2_Manager.getTrainDs(this.dataSet);
        this.testing_dataSet = Data2_Manager.getTestData(this.dataSet);
        this.testing_desired = Data2_Manager.getTestDs(this.dataSet);

        nodeBuilding(hidden);

    }

    public void nodeBuilding(int[] hidden) {

        int hNum = hidden.length;
        this.nodeLayer = new int[hNum + 2];
        this.nodeLayer[0] = training_dataSet.get(dataSet).size();
        for (int i = 1; i <= hNum + 1; i++) {

            if (i == hNum + 1) {
                this.nodeLayer[i] = training_desired.get(dataSet).size();
                this.error = new Double[this.nodeLayer[i]];
            } else
                this.nodeLayer[i] = hidden[i - 1];

        }
        this.nodeLayerNum = nodeLayer.length;

        this.node = new Double[nodeLayerNum][];
        this.local_gradient = new Double[nodeLayerNum][];

        for (int i = 0; i < nodeLayerNum; i++) {
            this.node[i] = new Double[nodeLayer[i]];
            this.local_gradient[i] = new Double[nodeLayer[i]];
        }

        this.weightOfLayer = new WeightGraph[nodeLayerNum - 1];
        this.changedWeight = new WeightGraph[nodeLayerNum - 1];

        for (int i = 0; i < weightOfLayer.length; i++) {
            weightOfLayer[i] = new WeightGraph(nodeLayer[i + 1], nodeLayer[i], true);
            changedWeight[i] = new WeightGraph(nodeLayer[i + 1], nodeLayer[i], false);
        }

        this.weightLayerNum = weightOfLayer.length;

    }

    public void training() {

        System.out.println("================= TRAINING "+(dataSet+1)+" =================");
        int n = 0;
        double avgError = 0.0;
        double correct = 0.0;
        int inNodeNum = nodeLayer[0];
        Random ranDataLine = new Random();
        int outputNum = nodeLayer[nodeLayerNum - 1];

        while (n < maxEpoch) {
            Double sum_error = 0.0;
            // insert input value to node
            for (int l = 0; l < training_dataSet.size(); l++) {
                int lineNum = ranDataLine.nextInt(training_dataSet.size());

                for (int i = 0; i < inNodeNum; i++) {
                    this.node[0][i] = training_dataSet.get(lineNum).get(i);
                }

                feedForward();
                errorCalculation(lineNum, true);
                backPropagation();

                Double[] outputArr = new Double[outputNum];

                Double[] desiredArr = new Double[outputNum];
                Double[] get = new Double[outputNum];
//                System.out.println("desired: ");
                for (int i = 0; i < outputNum; i++) {
                    desiredArr[i] = this.training_desired.get(lineNum).get(i);
//                    System.out.println(desiredArr[i]+"\t");
                }
                int maxIndex = 0;
//                System.out.println("Got: ");
                for (int i = 0; i < outputNum; i++) {
                    outputArr[i] = node[nodeLayerNum - 1][i];
                    if (i > 0 && outputArr[i] > outputArr[i - 1]) {
                        maxIndex = i;
                    }
                }

                for (int i = 0; i < outputNum; i++) {
                    if (i == maxIndex) {
                        get[i] = 1.0;
                    } else {
                        get[i] = 0.0;
                    }
//                    System.out.println(get[i] + "\t");
                }

                int neg_false, neg_true, pos_false, pos_true = 0;
                neg_false = neg_true = pos_false = pos_true;

                for (int i = 0; i < outputNum; i++) {
                    boolean equals = get[i].equals(desiredArr[i]);
                    if (equals && get[i].equals(1.0))
                        pos_true++;
                    else if(equals && get[i].equals(0.0))
                        neg_true++;
                    else if(!equals && get[i].equals(0.0))
                        pos_false++;
                    else if(!equals && get[i].equals(1.0))
                        neg_false++;
                }
            }
            avgError += correct / (training_dataSet.size() * outputNum) * 100;
            correct = 0.0;
            n++;
        }
        double correctness = avgError / n;
        System.out.println("% Correctness: " + correctness);
    }

    public void testing() {

        System.out.println("================= TESTING =================");
        int n = 0;
        double avgError;
        double correct = 0.0;
        int inNodeNum = nodeLayer[0];
        Random ranDataLine = new Random();
        int outputNum = nodeLayer[nodeLayerNum - 1];

        // insert input value to node
        for (int l = 0; l < testing_dataSet.size(); l++) {
            int lineNum = ranDataLine.nextInt(testing_dataSet.size());

            for (int i = 0; i < inNodeNum; i++) {
                this.node[0][i] = testing_dataSet.get(lineNum).get(i);
            }

            feedForward();
            errorCalculation(l, false);

            Double[] outputArr = new Double[outputNum];
            Double[] desiredArr = new Double[outputNum];
            Double[] get = new Double[outputNum];
//                System.out.println("desired: ");
            for (int i = 0; i < outputNum; i++) {
                desiredArr[i] = testing_desired.get(lineNum).get(i);
//                    System.out.println(desiredArr[i] + "\t");
            }

            int maxIndex = 0;
//                System.out.println("Got: ");
            for (int i = 0; i < outputNum; i++) {
                outputArr[i] = node[nodeLayerNum - 1][i];
                if (i > 0 && outputArr[i] > outputArr[i - 1]) {
                    maxIndex = i;
                }
            }

            for (int i = 0; i < outputNum; i++) {
                if (i == maxIndex) {
                    get[i] = 1.0;
                } else {
                    get[i] = 0.0;
                }
//                    System.out.println(get[i] + "\t");
            }

            for (int i = 0; i < outputNum; i++) {
                if (get[i].equals(desiredArr[i]))
                    correct++;
            }
        }
        avgError = correct / (testing_dataSet.size() * outputNum) * 100;

        System.out.println("% Correctness: " + (avgError));

    }

    public void feedForward(){


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
