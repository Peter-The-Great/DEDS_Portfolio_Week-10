using System;

public class NeuralNetwork
{
    private readonly int inputSize;
    private readonly int hiddenLayerSize;
    private readonly double[,] weightsInputToHidden;
    private readonly double[] weightsHiddenToOutput;
    private readonly double learningRate;

    public NeuralNetwork(int inputSize, int hiddenLayerSize, double learningRate)
    {
        this.inputSize = inputSize;
        this.hiddenLayerSize = hiddenLayerSize;
        this.learningRate = learningRate;

        var random = new Random();
        weightsInputToHidden = new double[inputSize, hiddenLayerSize];
        weightsHiddenToOutput = new double[hiddenLayerSize];
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                weightsInputToHidden[i, j] = random.NextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            weightsHiddenToOutput[i] = random.NextDouble() * 2 - 1;
        }
    }

    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    public double[] FeedForward(double[] inputs)
    {
        var hiddenLayerOutput = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputSize; j++)
            {
                sum += inputs[j] * weightsInputToHidden[j, i];
            }
            hiddenLayerOutput[i] = Sigmoid(sum);
        }

        double output = 0;
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            output += hiddenLayerOutput[i] * weightsHiddenToOutput[i];
        }
        output = Sigmoid(output);

        return new double[] { output };
    }

    public void Train(double[] inputs, double target)
    {
        var hiddenLayerOutput = new double[hiddenLayerSize];
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputSize; j++)
            {
                sum += inputs[j] * weightsInputToHidden[j, i];
            }
            hiddenLayerOutput[i] = Sigmoid(sum);
        }

        double output = 0;
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            output += hiddenLayerOutput[i] * weightsHiddenToOutput[i];
        }
        output = Sigmoid(output);

        double error = target - output;

        for (int i = 0; i < hiddenLayerSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                weightsInputToHidden[j, i] += error * learningRate * inputs[j];
            }
            weightsHiddenToOutput[i] += error * learningRate * hiddenLayerOutput[i];
        }
    }
}

public class Program
{
    public static void Main(string[] args)
    {
        var neuralNetwork = new NeuralNetwork(4, 3, 0.1);

        double[] inputs = { 0.1, 0.2, 0.3, 0.4 };
        double target = 0.5;
        
        for (int i = 0; i < 1000; i++)
        {
            neuralNetwork.Train(inputs, target);
        }

        double[] testInputs = { 0.5, 0.6, 0.7, 0.8 };
        var output = neuralNetwork.FeedForward(testInputs);

        Console.WriteLine("Output:" + output[0]);
    }
}