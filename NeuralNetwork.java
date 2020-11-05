import java.util.Random;

class BackpropagationExample {
    // 각 층의 노드의 개수
    private static final int INPUT_NEURONS = 12; // 4 → 6
    private static final int HIDDEN_NEURONS = 12; // 6 → 12
    private static final int SECOND_HIDDEN_NEURONS = 10;
    private static final int OUTPUT_NEURONS = 10; // 14 → 20

    // 학습률, 노이즈낀 인풋값들을 만들때 쓸 변수, 학습횟수
    private static final double LEARN_RATE = 0.4; // Rho. 0.2 → 0.4
    private static final int TRAINING_REPS = 21000; // 7000 → 21000(은닉층이 많아지니까 7천번의 횟수로는 정답률이 잘 안나옴)

    // 인풋에서 첫번째 은닉층으로 갈때 곱해줄 가중치와 Bias
    // Input to Hidden Weights (with Biases).
    private static double wih[][] = new double[INPUT_NEURONS + 1][HIDDEN_NEURONS];

    // 첫번째 은닉층에서 두번째 은닉층으로 갈때 곱해줄 가중치와 Bias
    // Hidden to Hidden Weights (with Biases).
    private static double whh[][] = new double[HIDDEN_NEURONS + 1][SECOND_HIDDEN_NEURONS];

    // 두번째 은닉층에서 출력층으로 갈때 곱해줄 가중치와 Bias
    // Hidden to Output Weights (with Biases).
    private static double who[][] = new double[SECOND_HIDDEN_NEURONS + 1][OUTPUT_NEURONS];

    // 각 층의 요소들을 담을 배열
    // Activations.
    private static double inputs[] = new double[INPUT_NEURONS];
    private static double hidden[] = new double[HIDDEN_NEURONS];
    private static double second_hidden[] = new double[SECOND_HIDDEN_NEURONS];
    private static double target[] = new double[OUTPUT_NEURONS];
    private static double actual[] = new double[OUTPUT_NEURONS];

    // 에러(가중치를 업데이트할때 사용)
    // Unit errors.
    private static double erro[] = new double[OUTPUT_NEURONS];
    private static double errsh[] = new double[SECOND_HIDDEN_NEURONS];
    private static double errh[] = new double[HIDDEN_NEURONS];

    // 샘플의 개수
    private static final int MAX_SAMPLES = 9; // 14 → 16

    // 학습 입력층 6 x 16개
    private static int trainInputs[][] = new int[][] { { 1, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0 },
            { 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0 }, { 1, 0, 1, 0, 3, 1, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1 },
            { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0 },
            { 1, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0 }, { 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 1 } };

    // 정담 16 x 20개
    private static int trainOutput[][] = new int[][] { { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 }, { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 }, };

    // 뉴럴 네트워크
    private static void NeuralNetwork() {
        int sample = 0;

        // 가중치 랜덤 할당
        assignRandomWeights();

        // Train the network.
        // 학습횟수만큼 학습 진행
        for (int epoch = 0; epoch < TRAINING_REPS; epoch++) {
            sample += 1;
            if (sample == MAX_SAMPLES) {
                sample = 0;
            }

            // 각 샘플의 요소들을 배열에 저장
            for (int i = 0; i < INPUT_NEURONS; i++) {
                inputs[i] = trainInputs[sample][i];
            } // i

            for (int i = 0; i < OUTPUT_NEURONS; i++) {
                target[i] = trainOutput[sample][i];
            } // i

            // 전파 알고리즘, 히든층들과 출력층이 업데이트 됨
            feedForward();

            // 라벨링된 정답과 비교한 후 역으로 에러를 계산하고 가중치를 업데이트함
            // wih, whh, who 업데이트됨(erro, errsh, errh 사용해서 가중치를 업데이트)
            backPropagate();

        } // epoch

        // 정답률 확인
        getTrainingStats();

        // 트레이닝 셋으로 결과값 확인
        System.out.println("\nTest network against original input:");
        testNetworkTraining();

        // 노이즈낀 트레이닝 셋으로 결과값 확인
        System.out.println("\nTest network against noisy input:");
        testNetworkWithNoise1();

        return;
    }

    private static void getTrainingStats() {
        double sum = 0.0;
        for (int i = 0; i < MAX_SAMPLES; i++) {
            for (int j = 0; j < INPUT_NEURONS; j++) {
                inputs[j] = trainInputs[i][j];
            } // j

            for (int j = 0; j < OUTPUT_NEURONS; j++) {
                target[j] = trainOutput[i][j];
            } // j

            feedForward();

            // 출력과 정답의 가장 높은 요소의 인덱스를 비교하여 같으면 정답으로 인정
            if (maximum(actual) == maximum(target)) {
                sum += 1;
            } else {
                System.out.println(inputs[0] + "\t" + inputs[1] + "\t" + inputs[2] + "\t" + inputs[3]);
                System.out.println(maximum(actual) + "\t" + maximum(target));
            }
        } // i

        // (정답 / 샘플 * 100)의 식으로 정답률 출력
        System.out.println("Network is " + ((double) sum / (double) MAX_SAMPLES * 100.0) + "% correct.");

        return;
    }

    private static void testNetworkTraining() {
        // This function simply tests the training vectors against network.
        for (int i = 0; i < MAX_SAMPLES; i++) {
            for (int j = 0; j < INPUT_NEURONS; j++) {
                inputs[j] = trainInputs[i][j];
            } // j

            feedForward();

            for (int j = 0; j < INPUT_NEURONS; j++) {
                System.out.print(inputs[j] + "\t");
            } // j

            System.out.print("Output: " + maximum(actual) + "\n");
        } // i

        return;
    }

    private static void testNetworkWithNoise1() {
        // This function adds a random fractional value to all the training
        // inputs greater than zero.

        double a[] = { 2, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0 };
        for (int i = 0; i < 12; i++) {
            inputs[i] = a[i];
        }

        feedForward();

        System.out.print("Output: " + maximum(actual) + "\n");

        return;
    }

    private static int maximum(final double[] vector) {
        // This function returns the index of the maximum of vector().
        int sel = 0;
        double max = vector[sel];

        for (int index = 0; index < OUTPUT_NEURONS; index++) {
            if (vector[index] > max) {
                max = vector[index];
                sel = index;
            }
        }
        return sel;
    }

    private static void feedForward() {
        double sum = 0.0;

        // Calculate input to hidden layer.
        for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
            sum = 0.0;
            for (int inp = 0; inp < INPUT_NEURONS; inp++) {
                sum += inputs[inp] * wih[inp][hid];
            } // inp

            sum += wih[INPUT_NEURONS][hid]; // Add in bias.
            hidden[hid] = sigmoid(sum);
        } // hid

        // Calculate hidden to hidden layer.
        for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
            sum = 0.0;
            for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
                sum += hidden[hid] * whh[hid][shid];
            } // inp

            sum += whh[HIDDEN_NEURONS][shid]; // Add in bias.
            second_hidden[shid] = sigmoid(sum);
        } // hid

        // Calculate the hidden to output layer.
        for (int out = 0; out < OUTPUT_NEURONS; out++) {
            sum = 0.0;
            for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
                sum += second_hidden[shid] * who[shid][out];
            } // hid

            sum += who[SECOND_HIDDEN_NEURONS][out]; // Add in bias.
            actual[out] = sigmoid(sum);
        } // out
        return;
    }

    private static void backPropagate() {
        // Calculate the output layer error (step 3 for output cell).
        for (int out = 0; out < OUTPUT_NEURONS; out++) {
            erro[out] = (target[out] - actual[out]) * sigmoidDerivative(actual[out]);
        }

        // Calculate the hidden layer error (step 3 for hidden cell).
        for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
            errsh[shid] = 0.0;
            for (int out = 0; out < OUTPUT_NEURONS; out++) {
                errsh[shid] += erro[out] * who[shid][out];
            }
            errsh[shid] *= sigmoidDerivative(second_hidden[shid]);
        }

        // Calculate the hidden layer error (step 3 for hidden cell).
        for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
            errh[hid] = 0.0;
            for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
                errh[hid] += errsh[shid] * whh[hid][shid];
            }
            errh[hid] *= sigmoidDerivative(hidden[hid]);
        }

        // Update the weights for the output layer (step 4).
        for (int out = 0; out < OUTPUT_NEURONS; out++) {
            for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
                who[shid][out] += (LEARN_RATE * erro[out] * second_hidden[shid]);
            } // hid
            who[SECOND_HIDDEN_NEURONS][out] += (LEARN_RATE * erro[out]); // Update the bias.
        } // out

        // Update the weights for the output layer (step 4).
        for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
            for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
                whh[hid][shid] += (LEARN_RATE * errsh[shid] * hidden[hid]);
            } // hid
            whh[HIDDEN_NEURONS][shid] += (LEARN_RATE * errsh[shid]); // Update the bias.
        } // out

        // Update the weights for the hidden layer (step 4).
        for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
            for (int inp = 0; inp < INPUT_NEURONS; inp++) {
                wih[inp][hid] += (LEARN_RATE * errh[hid] * inputs[inp]);
            } // inp
            wih[INPUT_NEURONS][hid] += (LEARN_RATE * errh[hid]); // Update the bias.
        } // hid
        return;
    }

    private static void assignRandomWeights() {
        for (int inp = 0; inp <= INPUT_NEURONS; inp++) // Do not subtract 1 here.
        {
            for (int hid = 0; hid < HIDDEN_NEURONS; hid++) {
                // Assign a random weight value between -0.5 and 0.5
                wih[inp][hid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp

        for (int hid = 0; hid <= HIDDEN_NEURONS; hid++) // Do not subtract 1 here.
        {
            for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) {
                // Assign a random weight value between -0.5 and 0.5
                whh[hid][shid] = new Random().nextDouble() - 0.5;
            } // hid
        } // inp

        for (int shid = 0; shid < SECOND_HIDDEN_NEURONS; shid++) // Do not subtract 1 here.
        {
            for (int out = 0; out < OUTPUT_NEURONS; out++) {
                // Assign a random weight value between -0.5 and 0.5
                who[shid][out] = new Random().nextDouble() - 0.5;
            } // out
        } // hid
        return;
    }

    private static double sigmoid(final double val) {
        return (1.0 / (1.0 + Math.exp(-val)));
    }

    private static double sigmoidDerivative(final double val) {
        return (val * (1.0 - val));
    }

    public static void main(String args[]) {
        NeuralNetwork();
    }
}