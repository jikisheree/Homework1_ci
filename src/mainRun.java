public class mainRun {
    public static void main(String[] args) {

        Double minError = 0.000001;
        int maxEpoch = 1000;
        Double bias = 1.0;
        Double learningRate = 0.04;
        Double mm = 0.1;
        int[] hidden = {7};
        Data1_Manager.getData();
        Data2_Manager.getData();
        int maxIteration = 2;

        for(int dataSet=0; dataSet<maxIteration; dataSet++) {
            NeuronNetwork nn = new NeuronNetwork(minError, learningRate, mm, maxEpoch, bias, hidden, dataSet);
            nn.training();
            nn.testing();
        }

//        for(int dataSet=0; dataSet<maxIteration; dataSet++) {
//
//            NeuronNetwork2 nn2 = new NeuronNetwork2(minError, learningRate, mm, maxEpoch, bias, hidden, dataSet);
//            nn2.training();
//            nn2.testing();
//        }

    }
}
