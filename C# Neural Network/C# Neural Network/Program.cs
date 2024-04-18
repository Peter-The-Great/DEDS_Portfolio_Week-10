/// <summary>
/// Een eenvoudige implementatie van een feedforward-neuraal netwerk in C#.
/// Die bestaat uit een inputlaag, een verborgen laag en een outputlaag. Waar gebruik van wordt gemaakt van de sigmoid-functie.
/// Daarnaast wordt er gebruik gemaakt van een trainingsmethode om het netwerk te trainen.
/// Die werkt door de gewichten van het netwerk aan te passen op basis van de fout tussen de voorspelde output en de werkelijke output.
/// </summary>
public class NeuralNetwork
{
    // Hier wordt de structuur van het neurale netwerk gedefinieerd
    // Met erin de inputSize, hiddenLayerSize, weightsInputToHidden, weightsHiddenToOutput en learningRate
    // Dit zijn de variabelen die nodig zijn om het neurale netwerk te kunnen trainen.
    // De random variabele wordt gebruikt om willekeurige getallen te genereren.
    private readonly int inputSize;

    private readonly Random random;
    private readonly int hiddenLayerSize;
    // Hier is weights een multidimensional array, omdat er meerdere inputs en meerdere hidden layers zijn.
    private readonly double[,] weightsInputToHidden;
    private readonly double[] weightsHiddenToOutput;
    private readonly double learningRate;

    public NeuralNetwork(int inputSize,int hiddenLayerSize, double learningRate)
    {
        // Stap 1: Defineer de structuur van het neurale netwerk,
        // door de inputSize, hiddenLayerSize en learningRate in te vullen
        this.hiddenLayerSize = hiddenLayerSize;

        // Stap 2: Initialiseer de gewichten van het netwerk willekeurig met een waarde tussen -1 en 1.
        // Wat we hier doen is de gewichten van de input naar de hidden layer
        // en van de hidden layer naar de output layer initialiseren.
        this.random = new Random();
        weightsInputToHidden = new double[inputSize, hiddenLayerSize];
        weightsHiddenToOutput = new double[hiddenLayerSize];
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenLayerSize; j++)
            {
                weightsInputToHidden[i, j] = random.NextDouble() * 2 - 1; // Willekeurige waarde tussen -1 en 1
            }
        }
        for (int i = 0; i < hiddenLayerSize; i++)
        {
            weightsHiddenToOutput[i] = random.NextDouble() * 2 - 1;
        }
    }

    // Dit is de methode die de output van het neurale netwerk berekent.
    /// <summary>
    /// Berekent de sigmoid van de gegeven waarde.
    /// </summary>
    /// <param name="x">De invoerwaarde.</param>
    /// <returns>De sigmoid van de invoerwaarde.</returns>
    private double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    // Stap 3: Implementeer de feedforward-methode
    /// <summary>
    /// Voert de feedforward-operatie uit van het neurale netwerk.
    /// </summary>
    /// <param name="inputs">De input waardes voor het neurale netwerk.</param>
    /// <returns>De output waardes voor het neurale netwerk.</returns>
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

    // Dit is de Train-methode die het neurale netwerk traint door de gewichten van het netwerk aan te passen.
    /// <summary>
    /// Traint het neurale netwerk met behulp van de gegeven input en het doel.
    /// </summary>
    /// <param name="inputs">De inputwaarden voor het netwerk.</param>
    /// <param name="target">Het gewenste doel voor de output.</param>
    public void Train(double[] inputs, double target)
    {
        // Stap 4: Bereken de fout tussen de voorspelde output en de werkelijke output. Met de fout kunnen we de gewichten aanpassen.
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

        // Stap 5: Pas de gewichten aan op basis van de error
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
        // Stap 6: Train het neurale netwerk met behulp van de Train-methode
        // Hier gaan we een eenvoudig voorbeeld gebruiken om het neurale netwerk te trainen
        var neuralNetwork = new NeuralNetwork(4, 3, 0.1);

        // Nu maken we een training op een datapunt.
        double[] inputs = { 0.1, 0.2, 0.3, 0.4 };
        double target = 0.5;
        
        Console.WriteLine("Training input:");
        foreach (var value in inputs)
        {
            Console.WriteLine(value);
        }
        
        for (int i = 0; i < 1000; i++)
        {
            neuralNetwork.Train(inputs, target);
        }

        // Nu gaan we nieuwe test datapunten toevoegen om te kijken hoe accuraat het netwerk is.
        double[] testInputs = { 0.5, 0.6, 0.7, 0.8 };
        var output = neuralNetwork.FeedForward(testInputs);
        
        Console.WriteLine("Test input:");
        foreach (var value in testInputs)
        {
            Console.WriteLine(value);
        }

        // Wat we hier uitprinten is de output van het neurale netwerk. Dat is de voorspelde waarde van hoe goed het netwerk is getraind.
        Console.WriteLine("Output:" + output[0]);
    }
}